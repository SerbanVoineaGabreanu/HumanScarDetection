First the dataset needs to be downloaded from https://github.com/mattgroh/fitzpatrick17k/tree/main?tab=readme-ov-file. Or the download script can be run.

The following folders will be present in the same directory as the scripts: Checkpoints, Conversations, DeployModel, Injury, LLMs, PerModelGraphs, ProcessedDataset, RawDataset, templates, ValidationGraphs

RawDataset is where the dataset should be saved by the downloader script.

Then the model must be trained with the training script and then deployed to the DeployModel folder

Next a local LLM has to be added to the LLMs directory (that also should be present in the same directory as the script), the script was tested with Gemma 3 27B Q4, the system prompt from the Investigator.py script may need to be modified if the LLM is changed.

The Manager.py script should be run once all that is completed, this gives the user the entry point to the program where they can upload their picture and get a classificatino result from the DL model. Next they can click next step and talk about their result with the LLM to figure out where it came from.

Once they are done they can click Next Step and then the LLM will produce the final story, as well as a tattoo idea.

(Disclaimer: This program should not be used for any real medical advice under any circumstances, it is not accurate enough and has not been thorougly tested.)
