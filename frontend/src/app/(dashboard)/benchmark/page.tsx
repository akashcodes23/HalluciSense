"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function BenchmarkRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/scientific");
  }, [router]);

  return null;
}
