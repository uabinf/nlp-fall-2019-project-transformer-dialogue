### Dialogue generation with Transformers

This project tries to generate dialogue utterances with an Encoder-Decoder built with a [transformer](http://jalammar.github.io/illustrated-transformer/) architecture. The dataset used for it are dialogues from [Opensubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php).


#### The dataset

The dataset used for the project can be downloaded [here](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz), and it contains 140 million utterances approximately. More dialogue datasets can be found in [this survey](https://breakend.github.io/DialogDatasets/).

The opensubtitles dataset is pretty dirty, with some metadata of the movies, comments not being part of the dialogues, descriptions, ...

We have used some [scripts] from Poly AI to clean and format the dataset. The script generates an output containing each of the utterances as shown below:


```
[Context]:
	Oh, my God, we killed her.
[Response]:
	Artificial intelligences cannot, by definition, be killed, Dr. Palmer.

Extra Contexts:
	[context/9]:
		So what are we waiting for?
	[context/8]:
		Nothing, it...
	[context/7]:
		It's just if...
	[context/6]:
		If we've underestimated the size of the artifact's data stream...
	[context/5]:
		We'll fry the ship's CPU and we'll all spend the rest of our lives stranded in the Temporal Zone.
	[context/4]:
		The ship's CPU has a name.
	[context/3]:
		Sorry, Gideon.
	[context/2]:
		Can we at least talk about this before you connect...
	[context/1]:
		Gideon?
	[context/0]:
		You still there?

Other features:
	[file_id]:
		lines-emk
```

Where the *context* are the previous dialogue sentences to the *response*. It is highly configurable and in our case we have generated it as :

* Just one context sentence. We just use the previous sentence and there is a huge performance gain when creating the files. 
* The script can generate the output as serialized Tensorflow format or JSON. We have created JSON outputs. 
* It also manages the dataset split. We created just one file with all the dialogues.

We have done some small modifications to the origial script, and this can be found in the respsitory (create_data.py)
