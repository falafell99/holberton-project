export default function RecommendationsTable({ recommendations }) {
  if (!recommendations.length) return null
  return (
    <>
      <table>
        <thead>
          <tr><th>Rank</th><th>Track</th><th>Score</th></tr>
        </thead>
        <tbody>
          {recommendations.map((rec, i) => (
            <tr key={rec.track_id}>
              <td>{i + 1}</td>
              <td>Track {rec.track_id}</td>
              <td>{rec.score.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Scores rank tracks within a model; they are not calibrated probabilities or comparable across models.</p>
    </>
  )
}
