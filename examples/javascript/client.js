/**
 * HalluciSense Node.js / JavaScript Client Example
 */

const BASE_URL = "http://localhost:8000/api/v1";

async function verifyHallucination(text) {
    const res = await fetch(`${BASE_URL}/verification/verify-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });
    
    if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
    }
    return await res.json();
}

// Example usage
verifyHallucination("Paris is the capital and most populous city of France.")
    .then(data => console.log("HalluciSense Verification Output:", data))
    .catch(err => console.error(err));
