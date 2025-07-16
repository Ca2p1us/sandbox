export async function submitRating(data: {
  user_id: string
  sound_id: string
  rating: number
}) {
  const res = await fetch('http://localhost:8000/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  return await res.json()
}
