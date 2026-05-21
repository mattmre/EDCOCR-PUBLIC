function buildWebSocketUrl(baseURL, pathname) {
  const url = new URL(pathname, baseURL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

async function collectWebSocketMessages(page, options) {
  return page.evaluate(async (config) => {
    const {
      apiKey,
      sendPingOnConnected,
      stopOnTypes,
      timeoutMs,
      wsUrl,
    } = config;

    return await new Promise((resolve, reject) => {
      const messages = [];
      let settled = false;
      let pingSent = false;
      const socket = new WebSocket(wsUrl);
      const timer = window.setTimeout(() => {
        rejectWith(
          new Error(`Timed out waiting for WebSocket messages from ${wsUrl}`));
      }, timeoutMs);

      const cleanup = => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        if (
          socket.readyState === WebSocket.CONNECTING
          || socket.readyState === WebSocket.OPEN
        ) {
          socket.close();
        }
      };

      const resolveWith = (value) => {
        cleanup();
        resolve(value);
      };

      const rejectWith = (error) => {
        cleanup();
        reject(error);
      };

      socket.addEventListener("open", => {
        if (apiKey) {
          socket.send(JSON.stringify({ type: "auth", api_key: apiKey }));
        }
      });

      socket.addEventListener("message", (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          rejectWith(
            new Error(`Expected JSON WebSocket payload but received: ${event.data}`));
          return;
        }

        messages.push(payload);
        if (
          sendPingOnConnected
          && !pingSent
          && payload.type === "connected"
        ) {
          pingSent = true;
          socket.send("ping");
        }

        if (stopOnTypes.includes(payload.type)) {
          resolveWith(messages);
        }
      });

      socket.addEventListener("error", => {
        if (!settled) {
          resolveWith(messages);
        }
      });

      socket.addEventListener("close", => {
        if (!settled) {
          resolveWith(messages);
        }
      });
    });
  }, options);
}

module.exports = {
  buildWebSocketUrl,
  collectWebSocketMessages,
};
