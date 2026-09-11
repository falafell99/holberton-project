// A quiet, deterministic waveform glyph rendered before a track name — five
// bars whose heights come from a hash of the track's own id, so the same
// track always draws the same "print," like a groove pattern.
function heights(seed) {
  const s = String(seed)
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
  const bars = []
  for (let i = 0; i < 5; i++) {
    h = (h * 1103515245 + 12345) >>> 0
    bars.push(30 + (h % 70))
  }
  return bars
}

export default function WaveformTick({ seed }) {
  return (
    <span className="tick" aria-hidden="true">
      {heights(seed).map((h, i) => (
        <span key={i} className="tick-bar" style={{ height: `${h}%` }} />
      ))}
    </span>
  )
}
