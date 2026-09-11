const OPTIONS = [
  { value: 'keep', label: 'Keep recorded event' },
  { value: 'full_listen', label: 'Full listen' },
  { value: 'like', label: 'Like' },
  { value: 'dislike', label: 'Dislike' },
  { value: 'short_listen', label: 'Short listen' },
]

export default function WhatIfPicker({ value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="what-if-select">Optional what-if change to the last event</label>
      <select id="what-if-select" value={value} onChange={(e) => onChange(e.target.value)}>
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      {value !== 'keep' && (
        <span className="caption">This is an explicitly edited what-if input. It does not change the recorded data or evaluation scores.</span>
      )}
    </div>
  )
}
