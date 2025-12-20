/**
 * Main Layout Component - RE-BMS v6.0
 */

import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';

export default function Layout() {
  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden transition-colors duration-300">
      {/* Header */}
      <Header />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar />

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6 bg-background transition-colors duration-300">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
