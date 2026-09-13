const ORDER = ['Most Popular', 'ItemKNN', 'Sequence-only GRU', 'NextBeat']

const FORMAT = {
  recall10: (v) => `${(v * 100).toFixed(2)}%`,
  ndcg10: (v) => v.toFixed(4),
  unconditional_recall10: (v) => `${(v * 100).toFixed(2)}%`,
  coverage10: (v) => `${(v * 100).toFixed(2)}%`,
  novelty10: (v) => v.toFixed(2),
  nfvr10: (v) => `${(v * 100).toFixed(3)}%`,
}

const LABEL = {
  recall10: 'Recall@10',
  ndcg10: 'NDCG@10',
  unconditional_recall10: 'Recall (all targets)',
  coverage10: 'Catalogue coverage',
  novelty10: 'Novelty (bits)',
  nfvr10: 'Dislike violations',
}

const COLUMNS = ['recall10', 'ndcg10', 'unconditional_recall10', 'coverage10', 'novelty10', 'nfvr10']

export default function MetricsPanel({ metrics, eligibleTargets, totalTargets }) {
  if (!metrics) return null
  return (
    <>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {COLUMNS.map((col) => <th key={col}>{LABEL[col]}</th>)}
          </tr>
        </thead>
        <tbody>
          {ORDER.map((name) => (
            <tr key={name}>
              <td>{name}</td>
              {COLUMNS.map((col) => <td key={col}>{FORMAT[col](metrics[name][col])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">
        {eligibleTargets.toLocaleString()} eligible in-catalogue targets; {totalTargets.toLocaleString()} total test targets. One recorded next listen per user.
      </p>
      <p className="caption">
        All models filter active dislikes from the full prior history, in both evaluation and live recommendations. Zero dislike violations reflect this rule. Repeat listens are allowed. Unknown test targets are excluded from conditional metrics and included as misses in unconditional Recall.
      </p>
    </>
  )
}
