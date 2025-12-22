/**
 * RE-BMS v6.1 - Desktop Web Application
 * Renewable Energy Bidding Management System
 */

import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard';
import SMPPrediction from './pages/SMPPrediction';
import Bidding from './pages/Bidding';
import Portfolio from './pages/Portfolio';
import Settlement from './pages/Settlement';
import Map from './pages/Map';
import Analysis from './pages/Analysis';
import KPXSimulation from './pages/KPXSimulation';
import RTMSimulation from './pages/RTMSimulation';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="smp" element={<SMPPrediction />} />
        <Route path="bidding" element={<Bidding />} />
        <Route path="kpx-simulation" element={<KPXSimulation />} />
        <Route path="rtm-simulation" element={<RTMSimulation />} />
        <Route path="portfolio" element={<Portfolio />} />
        <Route path="settlement" element={<Settlement />} />
        <Route path="map" element={<Map />} />
        <Route path="analysis" element={<Analysis />} />
      </Route>
    </Routes>
  );
}
