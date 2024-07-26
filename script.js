async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    const userMessage = userInput.value;
    if (userMessage.trim() === '') return;
    
    const userDiv = document.createElement('div');
    userDiv.textContent = `You: ${userMessage}`;
    chatBox.appendChild(userDiv);
    userInput.value = '';
    const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMessage }),
    });

    const data = await response.json();

   
    const botDiv = document.createElement('div');
    botDiv.textContent = `Bot: ${data.result}`;
    chatBox.appendChild(botDiv);

   
    chatBox.scrollTop = chatBox.scrollHeight;
}
