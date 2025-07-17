// frontend/src/lib/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; //FastAPI側のURL

export const postEvaluation = async (payload: {
  userId: string;
  rating: number;
  parameters: Record<string, any>;
}) => {
  return await axios.post(`${API_BASE_URL}/submit`, payload);
};
