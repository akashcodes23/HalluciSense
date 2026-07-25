'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

// Signup is now part of the unified /login page (sliding form)
export default function SignupRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace('/login'); }, [router]);
  return null;
}
