## Using OmniThinker

###Prerequisites
- Secrets: Set LM_KEY and SEARCHKEY as repository secrets in GitHub (Settings > Secrets and variables > Actions > Repository secrets).
- File Structure:
  - src/omni_think.py: Python script that implements OmniThinker and saves the article to results/article.md.
  - requirements.txt: Lists Python dependencies (e.g., for language models or APIs).
- Permissions: The default GITHUB_TOKEN has sufficient permissions to post comments and upload artifacts.

To generate an article based on a topic and the context of an issue or pull request, comment with:
@omnithinker: [your topic here]
- Replace `[your topic here]` with your desired topic.
- The workflow will generate an article and post it as a comment.
- If the article is too long, a link to download the full article will be provided.
- Ensure you provide a topic after the tag; otherwise, you'll receive a reminder.
