document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const questionInput = document.getElementById("question-input");
    const chatMessages = document.getElementById("chat-messages");
    const sendButton = document.getElementById("ask-button");

    if (!chatForm || !questionInput || !chatMessages) {
        console.error(
            "SmartDoc AI: Required HTML elements were not found."
        );
        return;
    }

    function addMessage(text, sender) {
        const message = document.createElement("div");
        message.className = `message ${sender}`;

        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = text;

        message.appendChild(content);
        chatMessages.appendChild(message);

        chatMessages.scrollTop = chatMessages.scrollHeight;

        return message;
    }

    function addSources(message, sources) {
        if (!Array.isArray(sources) || sources.length === 0) {
            return;
        }

        const details = document.createElement("details");
        details.className = "sources";

        const summary = document.createElement("summary");
        summary.textContent =
            `View source context (${sources.length})`;

        details.appendChild(summary);

        sources.forEach((source, index) => {
            const paragraph = document.createElement("p");

            const sourceText =
                typeof source === "string"
                    ? source
                    : (source.text || "");

            const pageNumber =
                typeof source === "object" && source !== null
                    ? source.page
                    : null;

            if (pageNumber !== null && pageNumber !== undefined) {
                paragraph.textContent =
                    `Source ${index + 1} | PDF Page ${pageNumber}\n${sourceText}`;
            } else {
                paragraph.textContent =
                    `Source ${index + 1}\n${sourceText}`;
            }

            details.appendChild(paragraph);
        });

        message.appendChild(details);
    }

    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const question = questionInput.value.trim();

        if (!question) {
            return;
        }

        addMessage(question, "user");
        questionInput.value = "";
        questionInput.focus();

        if (sendButton) {
            sendButton.disabled = true;
        }

        const loadingMessage = addMessage(
            "Thinking... Please wait.",
            "assistant"
        );

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            });

            const data = await response.json();
            console.log("API Response:", response.status, data);
            loadingMessage.remove();

            if (!response.ok) {
                addMessage(
                    data.error || "Something went wrong.",
                    "assistant"
                );
                return;
            }

            const answerMessage = addMessage(
                data.answer || "No answer was returned.",
                "assistant"
            );

            addSources(answerMessage, data.sources);

        } catch (error) {
            loadingMessage.remove();

            console.error("Chat request failed:", error);

            addMessage(
                "Unable to connect to SmartDoc AI. Please check the server.",
                "assistant"
            );

        } finally {
            if (sendButton) {
                sendButton.disabled = false;
            }
        }
    });
});
