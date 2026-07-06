import { useCallback, useEffect, useState } from "react";
import type { Stats } from "./api";
import { getStats } from "./api";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";

export default function App() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [connError, setConnError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await getStats();
      setStats(s);
      setConnError(null);
    } catch (e) {
      setConnError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="app">
      <Sidebar stats={stats} onChanged={refresh} />
      <div className="main-col">
        {connError && (
          <div className="conn-banner">
            Cannot reach the backend ({connError}). Start it with{" "}
            <code>uvicorn app.main:app</code> in the <code>backend</code> folder.
          </div>
        )}
        <Chat hasDocuments={(stats?.n_chunks ?? 0) > 0} onAsked={refresh} />
      </div>
    </div>
  );
}
