/**
 * HalluciSense JavaScript / TypeScript SDK Client
 * Official client library for HalluciSense Hallucination Verification API.
 */

class HalluciSenseClient {
  /**
   * @param {string} apiKey - HalluciSense API key (e.g. hs_live_...)
   * @param {string} baseUrl - Base API endpoint
   */
  constructor(apiKey, baseUrl = "http://localhost:8000/api/v1/pillar2") {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  /**
   * Verify text response for factual grounding.
   * @param {string} text 
   * @param {number} pillar1Prob 
   * @returns {Promise<Object>}
   */
  async verify(text, pillar1Prob = 0.50) {
    const response = await fetch(`${this.baseUrl}/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
        "User-Agent": "HalluciSense-JS-SDK/10.0.0"
      },
      body: JSON.stringify({ text, pillar1_probability: pillar1Prob })
    });

    if (!response.ok) {
      throw new Error(`HalluciSense API Error ${response.status}: ${await response.text()}`);
    }

    return await response.json();
  }
}

module.exports = { HalluciSenseClient };
