# Databricks notebook source
# MAGIC %md
# MAGIC ## Git - extra tips
# MAGIC This is just a short document where you can find a few extre tips on how to work with Git.
# MAGIC 
# MAGIC ### 1. CodeCommit
# MAGIC What you see in your instance is just your local repository. Sometimes, you might be interested in seeing what is in remote repository. For example, you have commited to your branch submission of your artificial use case and you would like to make sure that it safely arrived to the remote repository.
# MAGIC 
# MAGIC You can head over to AWS Management Console and here find a product called CodeCommit. It is nicely intuitive and you could examine our remote repository through various branches or commits.
# MAGIC 
# MAGIC ### 2. Reseting Git
# MAGIC 
# MAGIC Situation is as follows:
# MAGIC You have been working on notebooks during a virtual classrooms, maybe also started to work on use case and for some reason would like to have the newest materials from master on your own branch. 
# MAGIC 
# MAGIC Solution:
# MAGIC 
# MAGIC - Take everything which you have in your work folder to the root, so simply outside of work folder. 
# MAGIC - Switch to master “git checkout master”
# MAGIC - Fetch any recent change in remote “git fetch --all”
# MAGIC - Reset any changes in the local repository. Careful, this literally deletes everything inside the work folder, that is why all the notebooks and any material that the student worked on needs to be outside of work. “git reset --hard origin/master”
# MAGIC - Now you should see a fresh pull from master inside work.
# MAGIC - Switch to your original branch “git checkout student-original-branch”
# MAGIC - Bring the stuff from root back to the work folder and organize it as wished. Commit and push
# MAGIC 
# MAGIC This way you basically trick git into thinking that your local repository became empty and forced it load everything fresh from master. Afterwards, 
