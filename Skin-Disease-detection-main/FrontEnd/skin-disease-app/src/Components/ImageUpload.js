import { useState } from "react";
import axios from "axios";

export default function ImageUpload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await axios.post("http://localhost:5000/predict", formData);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error uploading image", error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-sky-100 to-indigo-200 p-6">
      <div className="w-full max-w-md bg-white p-6 rounded-2xl shadow-xl text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Skin Disease Detector</h2>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4 w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200"
        />

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="w-full h-64 object-cover rounded-lg border border-gray-300 shadow-sm mb-4"
          />
        )}

        <button
          onClick={handleUpload}
          disabled={!image || loading}
          className="w-full py-2 px-4 rounded-lg font-semibold text-white transition-colors duration-300
                     bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? "Processing..." : "Upload & Predict"}
        </button>

        {prediction && (
          <div className="mt-6 bg-gray-100 p-4 rounded-lg shadow-inner">
            <p className="text-lg font-semibold text-gray-700">Prediction:</p>
            <p className="text-xl text-indigo-600 font-bold">{prediction.prediction}</p>
            <p className="text-sm text-gray-500 mt-1">
              Confidence: <span className="font-medium">{prediction.confidence}%</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}


// import { useState } from "react";
// import axios from "axios";

// export default function ImageUpload() {
//   const [image, setImage] = useState(null);
//   const [preview, setPreview] = useState(null);
//   const [prediction, setPrediction] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleImageChange = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       setImage(file);
//       setPreview(URL.createObjectURL(file));
//       setPrediction(null);
//     }
//   };

//   const handleUpload = async () => {
//     if (!image) return;
//     setLoading(true);
//     const formData = new FormData();
//     formData.append("image", image);

//     try {
//       const response = await axios.post("http://localhost:5000/predict", formData);
//       setPrediction(response.data);
//     } catch (error) {
//       console.error("Error uploading image", error);
//     }
//     setLoading(false);
//   };

//   return (
//     <div className="flex flex-col items-center p-4 bg-white shadow-lg rounded-lg">
//       <input type="file" accept="image/*" onChange={handleImageChange} className="mb-4" />
//       {preview && <img src={preview} alt="Preview" className="w-64 h-64 object-cover rounded-lg mb-4" />}
//       <button
//         onClick={handleUpload}
//         disabled={!image || loading}
//         className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400"
//       >
//         {loading ? "Processing..." : "Upload & Predict"}
//       </button>
//       {prediction && (
//         <div className="mt-4 text-center">
//           <p className="text-lg font-semibold">Prediction: {prediction.prediction}</p>
//           <p className="text-sm text-gray-600">Confidence: {prediction.confidence}%</p>
//         </div>
//       )}
//     </div>
//   );
// }
