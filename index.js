

  var userBio = 'sample bio';
  var targetBios = {
    'John' : 'Adventurous and free-spirited bisexual looking for someone to share spontaneous road trips and cozy nights in. Passionate about photography, live music, and trying new foods. Let\'s explore the world and create beautiful memories together.',
    'Medina' : 'Nonbinary individual passionate about social justice and community building. Loves attending rallies, engaging in meaningful dialogue, and exploring local cultural events. Seeking a partner who shares a commitment to making a difference.',
    'Selena' : 'Lesbian who\'s all about living authentically and loving deeply. I enjoy cooking, gardening, and spending time at the beach. Seeking a genuine connection with someone who values honesty, kindness, and a sense of humor.',
    'Spencer' : 'Trans man with a love for nature and the great outdoors. When I\'m not hiking or camping, you\'ll find me volunteering at the local animal shelter. Seeking a kind-hearted partner to share adventures and quiet moments under the stars.',
    'Akul' : 'Gay man with a love for fitness and healthy living. Enjoys running, yoga, and meal prepping. Looking for someone who values wellness and is ready for both active dates and relaxing movie nights. Let\'s motivate each other to be our best selves.',
    'Jo' : 'Genderfluid writer and gamer. Passionate about storytelling, whether it\'s through novels, video games, or D&D campaigns. Looking for someone who enjoys both quiet nights in and epic fantasy worlds. Let\'s create our own story together.'
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
            targetBios.Selena,
            targetBios.Spencer,
            targetBios.Jo,
            targetBios.Akul
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
      var html = `<p>Your Bio: ${userBio}</p><br> <p> Name: ${Object.keys(targetBios)[i]} <br> Bio: ${targetBios[Object.keys(targetBios)[i]]} <br> Score: ${scores[i]} </p> <br>`;
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



