import os
from dotenv import load_dotenv
from praw import Reddit
from praw.models import MoreComments
from collections import deque


# load environment variables
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
username = os.getenv("USERNAME")
password = os.getenv("PASS")

# create reddit instance
reddit = Reddit(user_agent=True,
                     client_id=client_id, 
                     client_secret=client_secret,
                     username=username,
                     password=password)

url = 'https://www.reddit.com/r/interestingasfuck/comments/1eqjs5c/verne_troyer_posing_behind_one_of_shaquille/'
submission = reddit.submission(url=url)

# get all top-level comments on the submission
submission.comments.replace_more(limit=0)
for top_level_comment in submission.comments:
    print(top_level_comment.body)
    
# get all comments, level by level, via breadth-first-search, and write to file
with open('out.txt', 'w') as f:
    comment_queue = deque(submission.comments[:]) # seed queue with top level comments
    count = 0
    while comment_queue:
        comment = comment_queue.popleft()
        count += 1
        f.write(comment.body + '\n')
        comment_queue.extend(comment.replies)

print(f'submission.num_comments = {submission.num_comments}')
print(f'{count} comments counted using custom BFS')

# can also just use list method, which traverses via BFS
all_comments = submission.comments.list()
print(f'{len(all_comments)} comments found using list() method')
with open('outlist.txt', 'w') as f:
    for comment in all_comments:
        f.write(comment.body + '\n')