import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Send, MessageSquare, Copy, Check, RotateCcw, Target } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../../api/client'
import { useAppSelector, useAppDispatch } from '../../store'
import { sideClose, sideOptimisticAppend, sideOptimisticRollback } from '../../store/chatSlice'
import { copyToClipboard } from '../../utils/clipboard'
import MarkdownRenderer from '../../components/MarkdownRenderer'
import QueueStack from '../../components/QueueStack'
import BusySendButton, { useBusySendMode } from '../../components/BusySendButton'
import type { SideMessage } from '../../store/chatSlice'
import type { ChatMessage } from '../../types'

import { i18nT } from '../../i18n/t'
const MAX_QUESTION_BYTES = 32_768
// Max auto-grow height (px) for the side-question input before it scrolls.
const MAX_INPUT_H = 240
// How long the transient "queued instead" notice stays up. It describes a moment,
// not a state, so leaving it until the next submit would let it sit beside a later
// turn it has nothing to do with.
const NOTICE_TTL_MS = 8_000

function SideMessageBubble({ msg, isStreaming }: { msg: SideMessage; isStreaming: boolean }) {
  const [copied, setCopied] = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="rounded-md bg-accent/10 px-2.5 py-1.5 text-[13px] text-text whitespace-pre-wrap">
        {msg.steer && (
          <span className="flex items-center gap-1 text-[11px] font-medium text-accent mb-0.5">
            <Target size={11} />
            {i18nT('pages.chat.sideChat.steered')}
          </span>
        )}
        {msg.content}
      </div>
    )
  }

  if (msg.is_error) {
    return (
      <div className="rounded-md bg-danger/10 px-2.5 py-1.5 text-[13px] text-danger whitespace-pre-wrap">
        {msg.content}
      </div>
    )
  }

  return (
    <div className="group/side-msg rounded-md bg-bg-hover px-2.5 py-1.5 text-sm leading-relaxed text-text overflow-hidden" style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}>
      <MarkdownRenderer content={msg.content} streaming={isStreaming} />
      {!isStreaming && msg.content.length > 0 && (
        <div className="flex items-center gap-1 mt-0.5 opacity-0 transition-opacity group-hover/side-msg:opacity-100">
          <button
            className="text-muted hover:text-text p-0.5 rounded transition-colors"
            title={i18nT('pages.chat.sideChat.copy')}
            aria-label={copied ? i18nT('pages.chat.sideChat.copied') : i18nT('pages.chat.sideChat.copy')}
            onClick={() => {
              copyToClipboard(msg.content).then(() => {
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              }).catch(() => {})
            }}
          >
            {copied ? <Check size={13} className="text-ok" /> : <Copy size={13} />}
          </button>
        </div>
      )}
    </div>
  )
}

/** One composer submit. `steer` and `optimistic` are decided at submit time and
 *  carried along, so a mutation callback that runs later never re-derives them
 *  from state its own optimistic update already changed. */
type SideSubmit = { q: string; steer: boolean; optimistic: boolean }

/** Put `released` text back in the composer without discarding what is there.
 *
 *  Both texts are typed work: choosing either one destroys the other, and the
 *  released text has no other home (its card or its request is already gone), so
 *  it cannot be the one dropped. Appending keeps both and leaves the user to
 *  edit — visible, undoable by hand, and never a silent loss. */
function mergeDraft(prev: string, released: string): string {
  if (!prev.trim()) return released
  if (!released.trim()) return prev
  return [prev.trimEnd(), released].join('\n\n')
}

function relativeTime(iso: string): string | null {  const diff = Date.now() - new Date(iso).getTime()
  if (diff < 30 * 60_000) return null
  if (diff < 60 * 60_000) return `${Math.floor(diff / 60_000)}m`
  if (diff < 24 * 3600_000) return `${Math.floor(diff / 3600_000)}h`
  return `${Math.floor(diff / (24 * 3600_000))}d`
}

export default function SideChat({ slot }: { slot: string }) {
  const dispatch = useAppDispatch()
  const reduxSide = useAppSelector(s => s.chat.slotSide[slot])
  const parentTurnCount = useAppSelector(s =>
    s.chat.messages.filter(m => m.role === 'user' || m.role === 'assistant').length
  )
  const [draft, setDraft] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)
  // Transient, non-error feedback (e.g. a steer the server had to demote to a
  // queue entry). Kept apart from localError so it renders as a notice, not red.
  const [localNotice, setLocalNotice] = useState<string | null>(null)

  // Retire the notice on its own so it cannot outlive the moment it describes.
  useEffect(() => {
    if (!localNotice) return
    const t = setTimeout(() => setLocalNotice(null), NOTICE_TTL_MS)
    return () => clearTimeout(t)
  }, [localNotice])
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isNearBottomRef = useRef(true)

  const messages = reduxSide?.messages ?? []
  const isPending = reduxSide?.pending ?? false
  const queue = reduxSide?.queue ?? []
  const [busySendMode, setBusySendMode] = useBusySendMode()
  // A turn is in flight, so a submit can no longer just start one. Derived from
  // the same signal the thinking indicator uses, so the composer's affordance and
  // what the server will actually do can't disagree.
  const isBusy = isPending || (reduxSide?.streaming ?? false)

  const sendMutation = useMutation({
    mutationFn: async ({ q, steer }: SideSubmit) => {
      await api.sideOpen(slot)
      return api.sideTurn(slot, q, steer ? { steer: true } : undefined)
    },
    onMutate: ({ q, optimistic }: SideSubmit) => {
      setLocalError(null)
      setLocalNotice(null)
      if (optimistic) {
        const message: SideMessage = { role: 'user', content: q, ts: new Date().toISOString() }
        dispatch(sideOptimisticAppend({ slot, message }))
      }
      setDraft('')
    },
    onSuccess: res => {
      // A steer the server could not deliver becomes a queue entry. Say so:
      // otherwise the only signal that Steer turned into Queue is a card the user
      // has to notice on their own.
      if (res.demoted) setLocalNotice(i18nT('pages.chat.sideChat.steer_demoted_to_queue'))
    },
    onError: (_err, vars) => {
      // `optimistic` rides along in the vars rather than being recomputed here:
      // dispatching the bubble flips the side to busy, so re-deriving it in this
      // callback would read the post-submit state and skip the rollback.
      if (vars.optimistic) dispatch(sideOptimisticRollback(slot))
      // Nothing was accepted, so hand the text back — merged, not chosen: the
      // user may have started a new draft while the request was in flight.
      setDraft(prev => mergeDraft(prev, vars.q))
    },
  })

  // Queue ids whose cancel/edit is in flight. The card is only retired when the
  // server's frame lands, so without this a second click fires a duplicate that
  // races the first and returns 404 — reporting a failure for an action that
  // worked. Tracked per id rather than as one flag so two cards stay independent.
  const [pendingQueueIds, setPendingQueueIds] = useState<ReadonlySet<string>>(() => new Set())
  const markQueuePending = useCallback((queueId: string, pending: boolean) => {
    setPendingQueueIds(prev => {
      if (pending === prev.has(queueId)) return prev
      const next = new Set(prev)
      if (pending) next.add(queueId)
      else next.delete(queueId)
      return next
    })
  }, [])

  // Cancel and edit are SERVER-AUTHORITATIVE: the card is retired (or rewritten)
  // by the `chat.side_queue` frame the handler broadcasts, not optimistically.
  // A drain can dequeue the entry between render and click, so an optimistic
  // update would claim the text was cancelled while the turn it started is
  // already running — the one divergence a queue card must never show.
  const cancelQueued = useMutation({
    mutationFn: (queueId: string) => api.sideQueueCancel(slot, queueId),
    onMutate: (queueId: string) => { markQueuePending(queueId, true) },
    onSuccess: res => {
      // Merged, not replaced: once the server has let the entry go its text
      // exists nowhere else, and an in-progress draft is typed work too — so
      // neither may be the one discarded.
      setDraft(prev => mergeDraft(prev, res.content))
    },
    onError: () => {
      setLocalError(i18nT('pages.chat.sideChat.queue_cancel_failed'))
    },
    onSettled: (_d, _e, queueId) => { markQueuePending(queueId, false) },
  })

  const editQueued = useMutation({
    mutationFn: ({ queueId, content }: { queueId: string; content: string }) =>
      api.sideQueueEdit(slot, queueId, content),
    onMutate: ({ queueId }: { queueId: string; content: string }) => { markQueuePending(queueId, true) },
    onError: () => {
      setLocalError(i18nT('pages.chat.sideChat.queue_edit_failed'))
    },
    onSettled: (_d, _e, vars) => { markQueuePending(vars.queueId, false) },
  })

  /** Queue entries in the shape QueueStack renders, so the side panel and the
   *  main composer show one card design rather than two. */
  const queueCards = useMemo<ChatMessage[]>(
    () => queue.map(e => ({ role: 'queued', content: e.content, cls: 'msg msg-q', ts: e.ts, meta: { queueId: e.id } })),
    [queue]
  )

  const refreshMutation = useMutation({
    // local close is the source of truth — backend close errors are
    // intentionally not surfaced (the side state is gone locally either way).
    mutationFn: () => api.sideClose(slot),
    onMutate: () => {
      dispatch(sideClose(slot))
    },
  })

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }, [])

  const lastMessageContent = messages[messages.length - 1]?.content
  useEffect(() => {
    const el = scrollRef.current
    if (el && isNearBottomRef.current) {
      requestAnimationFrame(() => { el.scrollTop = el.scrollHeight })
    }
  }, [messages.length, lastMessageContent])

  // Select-to-Ask seed: when the user clicks "Ask" in the selection toolbar,
  // ChatPage opens this panel and fires a `side-seed` CustomEvent carrying the
  // selected text. Prefill the draft with the selection as a grounding
  // blockquote and focus the input so the user types their actual question
  // (which then fires sideOpen → sideTurn as usual). Isolated from main context.
  useEffect(() => {
    const onSeed = (e: Event) => {
      const detail = (e as CustomEvent<{ text?: string }>).detail
      const sel = detail?.text?.trim()
      if (!sel) return
      const quoted = sel.split('\n').map(line => `> ${line}`).join('\n')
      setDraft(prev => (prev.trim() ? `${prev.trimEnd()}\n\n${quoted}\n\n` : `${quoted}\n\n`))
      // Focus + place caret at the end so the user immediately types the question.
      requestAnimationFrame(() => {
        const el = textareaRef.current
        if (el) {
          el.focus()
          const len = el.value.length
          el.setSelectionRange(len, len)
          // Scroll to the top so the START of a long quote is visible (focusing
          // + caret-at-end scrolls to the bottom otherwise, hiding the quote).
          el.scrollTop = 0
        }
      })
    }
    window.addEventListener('side-seed', onSeed)
    return () => window.removeEventListener('side-seed', onSeed)
  }, [])

  // Auto-grow the input so a seeded multi-line quote (or a long typed question)
  // is fully visible instead of being clipped to the 2-row default. Grows with
  // content up to MAX_INPUT_H, then scrolls. The `min-h-[52px]` class floors it
  // at ~2 rows so an empty box keeps its original size.
  useLayoutEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_INPUT_H)}px`
  }, [draft])

  const send = useCallback(() => {
    const q = draft.trim()
    if (!q || sendMutation.isPending || !slot) return
    if (new Blob([q]).size > MAX_QUESTION_BYTES) {
      setLocalError(`Question too long (max ${MAX_QUESTION_BYTES.toLocaleString()} bytes)`)
      return
    }
    // While a turn runs, the split button decides: steer injects into it, queue
    // defers. From idle both collapse to "start a turn", so the flag is dropped.
    // An optimistic bubble belongs only to a turn this submit STARTS — a steer's
    // bubble has to land above the streaming answer and a queued one is a card,
    // so the server frame places both.
    const steer = isBusy && busySendMode === 'steer'
    sendMutation.mutate({ q, steer, optimistic: !isBusy })
  }, [draft, slot, sendMutation, isBusy, busySendMode])

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }, [send])

  const lastIdx = messages.length - 1
  const lastMsg = messages[lastIdx]
  const isStreaming = reduxSide?.streaming ?? false
  const isStreamingLast = lastMsg?.role === 'assistant' && isStreaming

  const sendErr = sendMutation.error
  const displayError = sendErr
    ? (sendErr instanceof Error ? sendErr.message : String(sendErr))
    : localError

  const turnsBehind = reduxSide ? parentTurnCount - reduxSide.openedAtTurnCount : 0
  const age = reduxSide?.createdAt ? relativeTime(reduxSide.createdAt) : null
  const showBanner = !!reduxSide && messages.length > 0
  const isStale = turnsBehind >= 10 || (reduxSide?.createdAt && Date.now() - new Date(reduxSide.createdAt).getTime() >= 4 * 3600_000)

  const handleRefresh = useCallback(() => {
    refreshMutation.mutate()
  }, [refreshMutation])

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {showBanner && (
        <div className={`flex items-center justify-between px-3 py-1.5 text-[12px] border-b border-border shrink-0 ${isStale ? 'bg-warning/10 text-warning' : 'bg-bg-hover/50 text-muted'}`}>
          <span className="italic">
            {i18nT('pages.chat.sideChat.context_from')} {i18nT('pages.chat.sideChat.turn', { count: turnsBehind })} {i18nT('pages.chat.sideChat.ago')}{age ? ` · ${age}` : ''}
          </span>
          <button
            onClick={() => void handleRefresh()}
            disabled={refreshMutation.isPending}
            className="flex items-center gap-1 text-[11px] font-medium text-accent hover:text-accent-hover disabled:opacity-50 bg-transparent border-none cursor-pointer disabled:cursor-not-allowed"
          >
            <RotateCcw size={11} className={refreshMutation.isPending ? 'animate-spin' : ''} />
            {i18nT('pages.chat.sideChat.refresh_context')}
          </button>
        </div>
      )}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted/30 gap-2 py-8">
            <span className="text-[24px]"><MessageSquare className="lucide-inline" /></span>
            <span className="text-[13px]">{i18nT('pages.chat.sideChat.ask_a_side_question_main_agent_keeps_working')}</span>
          </div>
        ) : (
          messages.map((m, i) => (
            <SideMessageBubble
              key={i}
              msg={m}
              isStreaming={i === lastIdx && isStreamingLast}
            />
          ))
        )}
        {isPending && lastMsg?.role === 'user' && (
          <div className="flex items-center gap-1.5 px-2.5 py-2 text-muted">
            <span className="flex gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" style={{ animationDelay: '300ms' }} />
            </span>
            <span className="text-[12px] streaming-indicator">{i18nT('pages.chat.sideChat.thinking')}</span>
          </div>
        )}
      </div>
      {displayError && (
        <div className="px-3 py-1 text-[12px] text-danger border-t border-border">{displayError}</div>
      )}
      {!displayError && localNotice && (
        <div className="px-3 py-1 text-[12px] text-muted border-t border-border" role="status">{localNotice}</div>
      )}
      {queueCards.length > 0 && (
        <div className="shrink-0 pt-1">
          <QueueStack
            messages={queueCards}
            fuseBelow={false}
            pendingIds={pendingQueueIds}
            onCancel={qid => { if (!pendingQueueIds.has(qid)) cancelQueued.mutate(qid) }}
            onEdit={(qid, content) => { if (!pendingQueueIds.has(qid)) editQueued.mutate({ queueId: qid, content }) }}
          />
        </div>
      )}
      <div className="border-t border-border p-2 flex items-end gap-2 shrink-0">
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label={i18nT('pages.chat.sideChat.ask_a_side_question')}
          placeholder={i18nT('pages.chat.sideChat.ask_a_side_question_2')}
          rows={2}
          style={{ maxHeight: MAX_INPUT_H }}
          className="flex-1 resize-none overflow-y-auto min-h-[52px] rounded-md border border-border bg-bg px-2 py-1.5 text-[13px] text-text focus:outline-none focus:border-accent disabled:opacity-60"
        />
        {isBusy ? (
          <BusySendButton
            mode={busySendMode}
            onModeChange={setBusySendMode}
            onFire={() => void send()}
            disabled={!draft.trim() || sendMutation.isPending}
          />
        ) : (
          <button
            onClick={() => void send()}
            disabled={sendMutation.isPending || !draft.trim()}
            className="shrink-0 px-2.5 py-1.5 rounded-md bg-accent text-accent-fg text-[12px] font-medium cursor-pointer hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed border-none"
            title={i18nT('pages.chat.sideChat.send')}
            aria-label={i18nT('pages.chat.sideChat.send')}
          >
            <Send size={13} />
          </button>
        )}
      </div>
    </div>
  )
}
