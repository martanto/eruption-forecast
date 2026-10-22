import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const [dark, setDark] = useState(
    () => document.documentElement.dataset.mode === "dark",
  );
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const followSystem = () => {
      try {
        if (localStorage.getItem("atlas-theme")) return;
      } catch { /* Storage is optional. */ }
      document.documentElement.dataset.mode = media.matches ? "dark" : "light";
      setDark(media.matches);
    };
    media.addEventListener("change", followSystem);
    return () => media.removeEventListener("change", followSystem);
  }, []);
  return (
    <Button variant="outline" className="theme-toggle" aria-label={`Switch to ${dark ? "light" : "dark"} theme`} onClick={() => {
      const mode = dark ? "light" : "dark";
      document.documentElement.dataset.mode = mode;
      setDark(!dark);
      try { localStorage.setItem("atlas-theme", mode); } catch { /* Storage is optional. */ }
    }}>
      {dark ? <Sun size={16} aria-hidden="true" /> : <Moon size={16} aria-hidden="true" />}
      {dark ? "Light" : "Dark"}
    </Button>
  );
}
