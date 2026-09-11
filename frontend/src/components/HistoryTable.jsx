export default function HistoryTable({ rows }) {
  return (
    <table>
      <thead>
        <tr><th>Track</th><th>Event</th><th>Played %</th></tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            <td>{row.track}</td>
            <td>{row.event}</td>
            <td>{row.played_percent ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
