import WaveformTick from './WaveformTick'

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
              <td className="rank">{String(i + 1).padStart(2, '0')}</td>
              <td>
                <span className="track-cell">
                  <WaveformTick seed={rec.track_id} />
                  Track {rec.track_id}
                </span>
              </td>
              <td className="num">{rec.score.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Scores rank tracks within a model; they are not calibrated probabilities or comparable across models.</p>
    </>
  )
}
