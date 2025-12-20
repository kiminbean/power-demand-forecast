/**
 * RE-BMS Mobile App v5.0
 * Mobile-optimized SMP Prediction, Bidding, Settlement
 */

import { Routes, Route, Navigate } from 'react-router-dom';
import MobileLayout from './components/MobileLayout';
import SMPPrediction from './pages/SMPPrediction';
import Bidding from './pages/Bidding';
import Settlement from './pages/Settlement';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<MobileLayout />}>
        <Route index element={<Navigate to="/smp" replace />} />
        <Route path="smp" element={<SMPPrediction />} />
        <Route path="bidding" element={<Bidding />} />
        <Route path="settlement" element={<Settlement />} />
      </Route>
    </Routes>
  );
}
