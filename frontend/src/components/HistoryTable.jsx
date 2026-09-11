import WaveformTick from './WaveformTick'

export default function HistoryTable({ rows }) {
  return (
    <table>
      <thead>
        <tr><th>Track</th><th>Event</th><th>Played %</th></tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            <td>
              <span className="track-cell">
                <WaveformTick seed={row.track} />
                {row.track}
              </span>
            </td>
            <td>{row.event}</td>
            <td className="num">{row.played_percent ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
