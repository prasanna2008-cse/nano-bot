const messagesEl = document.getElementById("messages");
const emptyEl = document.getElementById("empty");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const clearBtn = document.getElementById("clear");
const tempInput = document.getElementById("temperature");
const topkInput = document.getElementById("topk");
const tempValue = document.getElementById("tempValue");
const topkValue = document.getElementById("topkValue");

let history = [];
let busy = false;

tempInput.addEventListener("input", () => (tempValue.textContent = tempInput.value));
topkInput.addEventListener("input", () => (topkValue.textContent = topkInput.value));

function addRow(role, text) {
  if (emptyEl && emptyEl.parentNode) emptyEl.remove();
  const row = document.createElement("div");
  row.className = `row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "n";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.append(avatar, bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function showTyping() {
  const bubble = addRow("bot", "");
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  return bubble;
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

input.addEventListener("input", autoResize);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    autoResize();
    form.requestSubmit();
  });
});

clearBtn.addEventListener("click", () => {
  if (busy) return;
  history = [];
  messagesEl.innerHTML = "";
  if (emptyEl) messagesEl.appendChild(emptyEl);
  input.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;

  busy = true;
  sendBtn.disabled = true;
  input.value = "";
  autoResize();
  addRow("user", message);

  const bubble = showTyping();
  let reply = "";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        temperature: Number(tempInput.value),
        top_k: Number(topkInput.value),
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    bubble.classList.add("cursor");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      reply += decoder.decode(value, { stream: true });
      bubble.textContent = reply.replace(/^\s+/, "");
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    reply += decoder.decode();
    reply = reply.replace(/^\s+/, "").trim();
    bubble.textContent = reply || "(no reply)";
    bubble.classList.remove("cursor");

    history.push([message, reply]);
    history = history.slice(-8);
  } catch (error) {
    bubble.textContent =
      "The model could not reply. Make sure the server is still running.";
    bubble.classList.remove("cursor");
    console.error(error);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

input.focus();
