// frontend/src/App.tsx
import { useState } from 'react'
import { submitRating } from './api'

const sounds = ['sound1.mp3', 'sound2.mp3']

export default function App() {
  const [userId, setUserId] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [rating, setRating] = useState<number | null>(null)

  const handleSubmit = async () => {
    if (!userId || rating === null) {
      alert("IDと評価を入力してください")
      return
    }

    await submitRating({
      user_id: userId,
      sound_id: sounds[currentIndex],
      rating: rating
    })

    setRating(null)
    setCurrentIndex(prev => prev + 1)
  }

  if (currentIndex >= sounds.length) return <h2>実験完了！</h2>

  return (
    <div style={{ padding: '20px' }}>
      <h1>評価実験</h1>
      <input
        placeholder="被験者ID"
        value={userId}
        onChange={e => setUserId(e.target.value)}
      />
      <br /><br />
      <audio controls src={`/sounds/${sounds[currentIndex]}`} />
      <br /><br />
      <label>評価（1〜10）:</label>
      <select value={rating ?? ''} onChange={e => setRating(Number(e.target.value))}>
        <option value="">選択</option>
        {[...Array(10)].map((_, i) => (
          <option key={i+1} value={i+1}>{i+1}</option>
        ))}
      </select>
      <br /><br />
      <button onClick={handleSubmit}>送信</button>
    </div>
  )
}
