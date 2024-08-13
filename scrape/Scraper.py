"""
A RedditScraper class for retrieving comments from posts or entire subreddits
and writing to text files.
"""


import os
from typing import List
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


class RedditScraper():
    
    def __init__(self):
        # create reddit instance
        self.reddit = Reddit(user_agent=True,
                            client_id=client_id, 
                            client_secret=client_secret,
                            username=username,
                            password=password)
        # possible sorts to be used in searching a subreddit
        self.sorts = ['controversial', 'hot', 'new', 'rising', 'top']
        
    def get_comments(self, outfile: str, url: str = None, id: str = None) -> int:
        """Write all comments on the post at `url` to `outfile`.
        
        Traverses and writes the comments in a level-by-level manner
        i.e. top-level comments first, then second-level, etc.
        
        Returns the number of comments found.
        """
        submission = self.get_submission(url, id)
        submission.comments.replace_more(limit=None)  # replace all MoreComments objects, i.e. "load more comments" and "continue this thread" links
        # write all comments to outfile level by level via breadth-first-search
        with open(outfile, 'w') as f:
            f.write(submission.title + '\n') # write the submission title at the top of the outfile
            comment_queue = deque(submission.comments[:]) # seed queue with top level comments
            count = 0
            while comment_queue:
                comment = comment_queue.popleft()
                count += 1
                f.write(comment.body + '\n')
                comment_queue.extend(comment.replies)
        return count
    
    def get_top_level_comments(self, outfile: str, url: str = None, id: str = None) -> int:
        """Write all top-level comments on the submission to `outfile.`
        Returns the number of top-level comments found on the submission.
        """
        submission = self.get_submission(url, id)
        submission.comments.replace_more(limit=None)
        with open(outfile, 'w') as f:
            f.write(submission.title + '\n')
            count = 0
            for top_level_comment in submission.comments:
                count += 1
                f.write(top_level_comment.body + '\n')
        return count
    
    def get_submission(self, url: str = None, id: str = None):
        """Retrieve and return a reddit submission object.
        `id`: the submission id. The submission will be retrieved via id if this is provided.
        `url`: the submission url. Required if id is not provided.
        """
        if id:
            return self.reddit.submission(id)
        elif url:
            return self.reddit.submission(url=url)
        else:
            raise ValueError('Missing argument: a submission id or url must be provided.')
    
    def subreddit_info(self, sub: str):
        """Retrieve and print subreddit meta-data.
        `sub`: the subreddit display name.
        """
        subreddit = self.reddit.subreddit(sub)
        print(f'display name: {subreddit.display_name}')
        print(f'title: {subreddit.title}')
        print(f'description: {subreddit.description[:50]}...')
        
    def subreddit_iterate(self, sub: str, sort: str = 'hot', limit: int = 10) -> List[str]:
        """Iterate through a subreddit.
        `sub`: see above
        `sort`: the sorting method used to iterate through `sub`; must be one of `self.sorts`
        `limit`: the maximum number of submissions to iterate through
        Returns a list of the submission ids.
        """
        if sort not in self.sorts:
            raise ValueError(f'Invalid argument: {sort}. Must be one of {self.sorts}')
        # get the subreddit object
        subreddit = self.reddit.subreddit(sub)
        method = getattr(subreddit, sort)  # get the appropriate sort method
        # sort the subreddit and get the first `limit` submissions
        iterator = method(limit=limit)
        ids = [submission.id for submission in iterator]
        return ids
            

if __name__ == '__main__':
    scraper = RedditScraper()
    # url = 'https://www.reddit.com/r/interestingasfuck/comments/1eqjs5c/verne_troyer_posing_behind_one_of_shaquille/'
    # n = scraper.get_comments(url, 'test.txt')
    # print(f'{n} comments on {url}')
    sub = 'interestingasfuck'
    print(f'Exploring subreddit {sub}...')
    scraper.subreddit_info(sub)
    limit = 3
    for sort in scraper.sorts:
        print(f"Extracting comments from the first {limit} '{sort}' submissions in {sub}...")
        ids = scraper.subreddit_iterate(sub, sort, limit)
        for i, id in enumerate(ids):
            outfile = f'{sort}_{i + 1}.txt'
            print(f'{scraper.get_comments(outfile, id=id)} comments on post {id}')
    