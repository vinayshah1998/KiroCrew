import { useMutation, useQuery } from '@tanstack/react-query'
import { ApiError, api } from '../api/client'
import { i18nT } from '../i18n/t'
import { useAppDispatch, useAppSelector } from '../store'
import { updateSlot } from '../store/dashboardSlice'
import { addNotification } from '../store/notificationsSlice'
import type { ConfiguredChannelTarget, SessionLink } from '../types'
import { channelBrandLabel } from '../utils/channelOrigin'
import { ChannelBrandIcon } from './ChannelBrandIcon'
import { ContextMenuItem } from './ui/context-menu'
import { DropdownMenuItem } from './ui/dropdown-menu'

/**
 * One row per channel, and the row's LABEL is the action.
 *
 * There are exactly two states — connected and not — for every channel alike, so
 * this renders one flat list with no per-channel branches: `Disconnect from X`
 * when bound and delivering, `Connect to X` otherwise. Nothing here explains the
 * machinery. The role badge (Origin / Mirror / Two-way), the offline badge, the
 * reminder item, the release/stop-mirroring items and their confirms are all
 * gone, along with the vocabulary they carried.
 *
 * Disconnect means MUTE, never release: the binding survives, so the
 * conversation still resolves to this session and connecting again picks it back
 * up and catches it up. That is what lets one row carry both directions — a muted
 * channel and one that was never connected read identically, and the click that
 * connects either one does the right thing without the user knowing which it was.
 *
 * An `origin` link (the conversation a session was BORN in) gets no row: there is
 * no binding to connect or disconnect, and the sidebar already marks it from the
 * slot key.
 */

/** What one rendered row needs, whichever channel it belongs to. */
type ChannelRow = {
  key: string
  channel: string
  label: string
  connected: boolean
  disabledReason: string
  toggle: () => void
}

export default function LinkedSurfacesSection({ slotKey, variant }: {
  slotKey: string
  variant: 'dropdown' | 'context'
}) {
  const Item = variant === 'context' ? ContextMenuItem : DropdownMenuItem
  const dispatch = useAppDispatch()
  const slot = useAppSelector(s => s.dashboard.slots.find(x => x.key === slotKey))
  const wireLinks = slot?.links ?? []
  // No synthesized Slack row. The wire now emits a Slack row on exactly the
  // condition it reports `slack_linked`, and the row is what carries `paused` — a
  // row invented here could not know the mute, so it rendered a MUTED thread as
  // connected. Trusting the wire is what keeps the two from disagreeing;
  // `TestASlackRowAlwaysAccompaniesSlackLinked` pins the backend side.
  const links: SessionLink[] = wireLinks
  const slackLink = links.find(link => link.channel === 'slack')
  // Every non-Slack binding this session holds. `origin` is not a binding — it is
  // where the session came from — so it is never a row.
  const mirrorLinks = links.filter(link => link.channel !== 'slack' && link.direction !== 'origin')

  const { data: targets } = useQuery({
    queryKey: ['channel-targets'],
    queryFn: () => api.channelTargets().then(result => (
      Array.isArray(result) ? result as ConfiguredChannelTarget[] : []
    )),
    refetchInterval: 30_000,
  })

  // The Slack row's label must NOT vary with connection state, or it stops
  // reading as one row with two states: the wire's link label is "Slack" while the
  // picker's target label is "Slack · Direct Message". Take the brand label, which
  // is the one place the repo keeps channel brand names (and documents why they are
  // not catalog entries), so both states read the same.
  const slackLabel = channelBrandLabel('slack')
    || slackLink?.label
    || (targets ?? []).find(target => target.channel_type === 'slack')?.label
    || ''

  const notify = (kind: 'success' | 'error', title: string) => {
    dispatch(addNotification({ ts: String(Date.now()), title, body: '', kind }))
  }
  const failure = (e: unknown) => (
    e instanceof Error && e.message
      ? e.message
      : i18nT('components.linkedSurfacesSection.unknown_error')
  )

  /** Rewrite one channel's row in place. Disconnect RETAINS the binding, so
   *  dropping the row would tell the user the opposite of what happened and
   *  would strip the control they need to connect again. */
  const withPaused = (channel: string, paused: boolean) => links.map(link => (
    link.channel === channel ? { ...link, paused } : link
  ))
  /**
   * Optimistic UPSERT of a channel's row. A first-ever connect has no row to
   * rewrite, and the component no longer invents one from `slack_linked` (an
   * invented row cannot know `paused`, so a muted thread rendered as connected).
   * Here we just performed the connect, so the state is known rather than guessed:
   * `paused: false` is a fact, not an assumption. The server's next slots push
   * replaces this with the authoritative row.
   */
  const withRow = (channel: string, label: string, target: string) => (
    links.some(link => link.channel === channel)
      ? withPaused(channel, false)
      : [...links, {
        channel, label, target, direction: 'out' as const, live: true, paused: false,
      }]
  )

  // Every mutation notifies on failure. None of them has a visible result outside
  // this menu — a disconnect is silent in the conversation, and a connect's
  // catch-up lands where the user is not looking — so a silent failure would
  // leave them believing the state flipped when it did not. Success needs no
  // toast: the verb flipping is the confirmation.
  const connectSlack = useMutation({
    mutationFn: (channel: string | undefined) => api.slackLink(slotKey, channel),
    onSuccess: (r) => {
      if (!r?.ok) return
      dispatch(updateSlot({
        key: slotKey,
        links: withRow('slack', slackLabel, r.channel ?? ''),
        slack_linked: true,
        slack_channel: r.channel,
        slack_thread_ts: r.thread_ts,
      }))
    },
    onError: (e) => notify('error', i18nT('components.linkedSurfacesSection.connect_failed', {
      label: slackLabel, reason: failure(e),
    })),
  })
  const disconnectSlack = useMutation({
    mutationFn: () => api.pauseSlack(slotKey),
    onSuccess: () => dispatch(updateSlot({ key: slotKey, links: withPaused('slack', true) })),
    onError: (e) => notify('error', i18nT('components.linkedSurfacesSection.disconnect_failed', {
      label: slackLabel, reason: failure(e),
    })),
  })
  const connectMirror = useMutation({
    mutationFn: ({ target, channel, confirm }: {
      target: ConfiguredChannelTarget | null
      channel: string
      confirm?: boolean
    }) => (
      target
        ? api.linkMirror(slotKey, target.channel_type, target.target_id, confirm)
        : api.reconnectMirror(slotKey, channel)
    ),
    onSuccess: (result, { target, channel }) => {
      if (!result?.ok) return
      // A fresh link needs its row minted; a reconnect only needs the mute lifted.
      dispatch(updateSlot({
        key: slotKey,
        links: target
          ? [
            ...links.filter(link => (
              link.direction === 'origin'
              || link.channel === 'slack'
              || link.channel !== target.channel_type
            )),
            {
              channel: target.channel_type,
              label: target.label,
              target: result.conversation_id || target.target_id,
              direction: 'out',
              live: true,
            },
          ]
          : withPaused(channel, false),
      }))
    },
    onError: (e, { target, channel, confirm }) => {
      // 409 conversation_occupied: another session holds this conversation. A
      // conversation cannot host two (there are no threads to scope them to), so
      // taking it means disconnecting the other — the user's call, asked once.
      // Match the STATUS, not the prose: `friendlyErrText` unwraps the body's
      // `error` field and drops `code`, so the message the client actually
      // produces contains neither "conversation_occupied" nor "409".
      const occupied = e instanceof ApiError && e.status === 409
      if (occupied && target && !confirm) {
        if (window.confirm(i18nT('components.linkedSurfacesSection.confirm_takeover', {
          label: target.label,
        }))) {
          connectMirror.mutate({ target, channel, confirm: true })
        }
        return
      }
      notify('error', i18nT('components.linkedSurfacesSection.connect_failed', {
        label: target?.label ?? channel, reason: failure(e),
      }))
    },
  })
  const disconnectMirror = useMutation({
    mutationFn: (channel: string) => api.pauseMirror(slotKey, channel),
    onSuccess: (_result, channel) => dispatch(updateSlot({
      key: slotKey, links: withPaused(channel, true),
    })),
    onError: (e, channel) => notify(
      'error',
      i18nT('components.linkedSurfacesSection.disconnect_failed', {
        label: links.find(link => link.channel === channel)?.label ?? channel,
        reason: failure(e),
      }),
    ),
  })

  const rows: ChannelRow[] = []

  // Slack, when it is linked or offered. A muted link reconnects with NO channel
  // argument so the endpoint reuses its existing thread; only a session that has
  // never linked passes the picker's target and mints one.
  const slackTarget = (targets ?? []).find(target => target.channel_type === 'slack')
  if (slackLink || slackTarget) {
    const connected = slackLink != null && !slackLink.paused
    rows.push({
      key: 'slack',
      channel: 'slack',
      label: slackLabel,
      connected,
      disabledReason: !slackLink && slackTarget && !slackTarget.available
        ? slackTarget.unavailable_reason || i18nT('components.linkedSurfacesSection.unavailable')
        : '',
      toggle: () => {
        if (connected) {
          if (!disconnectSlack.isPending) disconnectSlack.mutate()
        } else if (!connectSlack.isPending) {
          connectSlack.mutate(slackLink ? undefined : slackTarget?.target_id)
        }
      },
    })
  }

  // Every non-Slack binding, live or muted — one row each.
  for (const link of mirrorLinks) {
    const connected = !link.paused
    rows.push({
      key: `${link.channel}:${link.target}`,
      channel: link.channel,
      label: link.label,
      connected,
      disabledReason: '',
      toggle: () => {
        if (connected) {
          if (!disconnectMirror.isPending) disconnectMirror.mutate(link.channel)
        } else if (!connectMirror.isPending) {
          connectMirror.mutate({ target: null, channel: link.channel })
        }
      },
    })
  }

  // Offers for channels this session does not already hold. Slack has its own row
  // above; a channel already bound has its row instead of an offer, so connecting
  // a second conversation on the same channel is not offered — one binding per
  // channel type.
  const boundChannels = new Set(mirrorLinks.map(link => link.channel))
  const offers = (targets ?? []).filter(
    target => target.channel_type !== 'slack' && !boundChannels.has(target.channel_type),
  )
  for (const target of offers) {
    rows.push({
      key: `${target.channel_type}:${target.target_id}`,
      channel: target.channel_type,
      label: target.label,
      connected: false,
      disabledReason: target.available
        ? ''
        : target.unavailable_reason || i18nT('components.linkedSurfacesSection.unavailable'),
      toggle: () => {
        if (!connectMirror.isPending) {
          connectMirror.mutate({ target, channel: target.channel_type })
        }
      },
    })
  }

  return (
    <>
      {rows.map(row => (
        <Item
          key={row.key}
          aria-disabled={row.disabledReason ? true : undefined}
          className={row.disabledReason ? 'opacity-60' : undefined}
          // The row's ONLY tooltip, and only when the channel cannot be linked at
          // all: a broken config is a fact the user cannot otherwise see. The
          // retained-binding behaviour is deliberately never explained.
          title={row.disabledReason || undefined}
          onSelect={(event) => {
            // Never close the menu: the row IS the state display, so the user has
            // to stay to see the verb flip. A menu that closes on click reads as
            // "nothing happened".
            event.preventDefault()
            if (row.disabledReason) {
              notify('error', row.disabledReason)
              return
            }
            row.toggle()
          }}
        >
          <ChannelBrandIcon channel={row.channel} size={13} />
          <span className="truncate">
            {row.connected
              ? i18nT('components.linkedSurfacesSection.disconnect_from', { label: row.label })
              : i18nT('components.linkedSurfacesSection.connect_to', { label: row.label })}
          </span>
        </Item>
      ))}
    </>
  )
}
