export default function ModelSelect({ models, defaultModel, value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="model-select">Model</label>
      <select id="model-select" value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
        {models.map((name) => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>
      <span className="caption">Default model selected by validation NDCG@10: {defaultModel}.</span>
    </div>
  )
}
