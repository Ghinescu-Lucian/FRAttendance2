import { useEffect, useState } from "react";
import { FaceEnrollmentPage } from "./FaceEnrollmentPage";
import { FaceRecognitionPage } from "./FaceRecognitionPage";

type Page = "enrollment" | "recognition";

function getInitialPage(): Page {
  return window.location.hash === "#recognition" ? "recognition" : "enrollment";
}

export function App() {
  const [page, setPage] = useState<Page>(getInitialPage);

  useEffect(() => {
    const onHashChange = () => setPage(getInitialPage());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function navigate(nextPage: Page): void {
    window.location.hash = nextPage === "recognition" ? "recognition" : "enrollment";
    setPage(nextPage);
  }

  return (
    <>
      <nav className="app-nav">
        <button
          type="button"
          className={page === "enrollment" ? "nav-btn active" : "nav-btn"}
          onClick={() => navigate("enrollment")}
        >
          Enrollment
        </button>
        <button
          type="button"
          className={page === "recognition" ? "nav-btn active" : "nav-btn"}
          onClick={() => navigate("recognition")}
        >
          Recognition
        </button>
      </nav>

      {page === "enrollment" ? <FaceEnrollmentPage /> : <FaceRecognitionPage />}
    </>
  );
}
