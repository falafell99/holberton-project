export default function UserSelect({ users, value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="user-select">Anonymous user</label>
      <select id="user-select" value={value ?? ''} onChange={(e) => onChange(Number(e.target.value))}>
        {users.map((uid) => (
          <option key={uid} value={uid}>{uid}</option>
        ))}
      </select>
    </div>
  )
}
