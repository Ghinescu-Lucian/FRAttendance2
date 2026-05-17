import React from "react";
import { createRoot } from "react-dom/client";
import { FaceEnrollmentPage } from "./presentation/FaceEnrollmentPage";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element in index.html");

createRoot(root).render(
  <React.StrictMode>
    <FaceEnrollmentPage />
  </React.StrictMode>,
);
