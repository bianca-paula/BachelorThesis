alert("Pop up");

// !(function () {
//   let e = document.createElement("script"),
//     t = document.head || document.getElementsByTagName("head")[0];
//   (e.src =
//     "https://cdn.jsdelivr.net/npm/rasa-webchat@1.x.x/lib/index.js"),
//     // Replace 1.x.x with the version that you want
//     (e.async = !0),
//     (e.onload = () => {
//       window.WebChat.default(
//         {
//           customData: { language: "en" },
//           socketUrl: "http://localhost:5005/webhook",
//           // add other props here
//         },
//         null
//       );
//     }),
//     t.insertBefore(e, t.firstChild);
// })();




function addChat() {
  console.log("aaaaaaaaaaaaaaaaaaaaa");
  var chatroom = new window.Chatroom({
    host: "http://localhost:5005",
    title: "Chat with Mike",
    container: document.querySelector(".chat-container"),
    welcomeMessage: "Hi, I am Mike. How may I help you?",
    speechRecognition: "en-US",
    voiceLang: "en-US"
  });
  chatroom.openChat();

}
