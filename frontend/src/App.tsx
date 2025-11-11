import PECChatbot from './components/PECChatbot';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Pakistan Engineering Council
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Your trusted partner for engineering excellence and professional development.
          </p>

          <div className="bg-white rounded-lg shadow-md p-8 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Our Services</h2>
            <ul className="space-y-3 text-gray-700">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Engineer Registration and Professional Licensing</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Firm Registration and Consulting Services</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Continuing Professional Development (CPD) Programs</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Project Approvals and Technical Evaluations</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Professional Engineering Examinations</span>
              </li>
            </ul>
          </div>

          <div className="bg-blue-50 rounded-lg shadow-md p-8 border border-blue-100">
            <h2 className="text-2xl font-semibold text-blue-900 mb-3">Need Assistance?</h2>
            <p className="text-gray-700">
              Click the chat icon in the bottom-right corner to talk with our PEC Assistant.
              Get instant help with registration, licensing, CPD, and all PEC services.
            </p>
          </div>
        </div>
      </div>

      <PECChatbot />
    </div>
  );
}

export default App;
