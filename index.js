

  var userBio = 'sample bio';
  var targetBios = {
    'John' : 'Actor and playwright, always involved in local productions. Passionate about all things theater. Let\'s enjoy a show and discuss the arts!',
    'Medina' : 'Globe-trotter who\'s always on the go. Loves discovering new cultures and cuisines. Seeking a partner in adventure. Next stop: Anywhere!',
    'Selena' : 'Collector of board games and host of epic game nights. Looking for someone to challenge me in Catan or Ticket to Ride. Roll the dice with me!'
  };
  console.log(targetBios.John);
  var scores = [];
  function getUserInput() {
    userBio = document.getElementById('inBio').value;

    if(userBio.length < 3) { //bio must be longer than 3 chars
        console.log('bio too short!');
    } else {
      console.log(userBio);
      use.loadQnA().then(model => {
        console.log("loading");
        const input = {
          queries: [userBio],
          responses: [
            targetBios.John,
            targetBios.Medina,
            targetBios.Selena
          ]
        };
        
        //calculate embeddings for inputs, output scores for q/r pairs
        const embeddings = model.embed(input);
        scores = tf.matMul(embeddings['queryEmbedding'],
            embeddings['responseEmbedding'], false, true).dataSync();
            console.log(scores);
            renderScores(scores);
      });
      
    }
  };
 
  //display socres on html
  function renderScores(scores) {
    console.log('rendering');
    //testing
    for(var i = 0; i < scores.length; i++) {
      var div = document.getElementById("resultHolder");
      var html = `<p> Name: ${Object.keys(targetBios)[i]} <br> Bio: ${targetBios[Object.keys(targetBios)[i]]} <br> Score: ${scores[i]} </p> <br>`;
      div.insertAdjacentHTML('beforeend', html);
    }

  };

/* From TensorFlow's documentation
  use.loadQnA().then(model => {
    // Embed a dictionary of a query and responses. The input to the embed method
    // needs to be in following format:
    // {
    //   queries: string[];
    //   responses: Response[];
    // }
    // queries is an array of question strings
    // responses is an array of following structure:
    // {
    //   response: string;
    //   context?: string;
    // }
    // context is optional, it provides the context string of the answer.

    /*
      * The output of the embed method is an object with two keys:
      * {
      *   queryEmbedding: tf.Tensor;
      *   responseEmbedding: tf.Tensor;
      * }
      * queryEmbedding is a tensor containing embeddings for all queries.
      * responseEmbedding is a tensor containing embeddings for all answers.
      * You can call `arraySync()` to retrieve the values of the tensor.
  }); */



