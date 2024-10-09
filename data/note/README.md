## Main data for the competition
The legal corpus is in 'law.json' file. Training data for Task 1 (Legal Document Retrieval) and Task 2 (Legal Question Answering) is divided into two files: expert-verified data (please check file 'train.json') and unverified data (please check file 'unverified_train.json'). Expert-verified data has been annotated and confirmed by at least two experts for accuracy, while unverified data has annotations but lacks expert confirmation.

There are more unverified data samples than expert-verified ones (430 compared to 100), but they may contain noise. It is optional to use the unverified data to train your models.

## Additional data for Task 1 and Task 2
We provide the training data of ALQAC 2022 (both tasks) in the folder 'additional_data/ALQAC_2022_training_data'. It is optional to use this data to train your models.

## Additional data for Task 1
The additional data for Task 1, which is sponsored by Zalo, is in the folder 'additional_data/zalo'. It is optional to use this data to train your models.

## Note
You are free to crawl Internet data as additional data. There are no limitations on the use of externally sourced data. Participants may use any large language models (LLMs) and pre-trained models. However, all participating teams are required to submit their source code for verification.


## Leaderboard: Evaluate your system with the public test
This year, we provide a leaderboard for the teams to submit their systems' predictions for the public test (please check file 'public_test.json'). The public test only contains Yes/No questions (Câu hỏi Đúng/Sai) and Multiple-choice questions (Câu hỏi trắc nghiệm) for auto evaluation. Note that the public test leaderboard serves as a reference and may not accurately reflect the actual performance of participating systems. The final results will be validated using the private test set. 

The leaderboard website: https://eval.ai/web/challenges/challenge-page/2294/overview
Please register a team for the ALQAC 2024 competition on this website (it should be the same as the team name you registered in the email) and let us know, so that we will allow your team to submit on the leaderboard.

The deadline to submit to public test leaderboard is as follows:
- Task 1: June 03, 2024 - June 17, 2024
- Task 2: June 18, 2024 - June 31, 2024

Upon the conclusion of the Task 1 submission deadline, gold labels for Task 1 public test will be distributed and can be utilized for Task 2 public test.