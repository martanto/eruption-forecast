import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";

export function usePolling<T>(path: string, interval = 2000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try {
      setData(await api<T>(path));
      setError("");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Connection failed");
    }
  }, [path]);
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), interval);
    return () => clearInterval(timer);
  }, [refresh, interval]);
  return { data, error, refresh };
}
