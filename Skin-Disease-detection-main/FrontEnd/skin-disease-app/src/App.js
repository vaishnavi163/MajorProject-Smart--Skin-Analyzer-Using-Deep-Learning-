import ImageUpload from "./Components/ImageUpload";

function App() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6">Smart Skin Analyzer</h1>
      <ImageUpload />
    </div>
  );
}

export default App;
