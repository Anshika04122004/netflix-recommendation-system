import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
});

export const fetchHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const fetchModelInfo = async () => {
  const response = await api.get('/model-info');
  return response.data;
};

export const fetchAnalytics = async () => {
  const response = await api.get('/analytics');
  return response.data;
};

export const predictRating = async (contentData) => {
  const response = await api.post('/predict', contentData);
  return response.data;
};

export default api;
