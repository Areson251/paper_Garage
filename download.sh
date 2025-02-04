
for file in $(ssh -i ~/.ssh/mlspace__private_key.txt -p 2222 afilatov-paper-2.ai0001053-00016@ssh-sr004-jupyter.ai.cloud.ru 'ls -p ignashin/Garage/my_Garage/Garage/augmentations/generation_6/bboxes/ | grep -v / | head -n 50'); do
    scp -i ~/.ssh/mlspace__private_key.txt -P 2222 afilatov-paper-2.ai0001053-00016@ssh-sr004-jupyter.ai.cloud.ru:"ignashin/Garage/my_Garage/Garage/augmentations/generation_6/bboxes/$file" /Users/ulyanaizmesteva/VSprojects/paper_Garage/generation_6/bboxes
done


