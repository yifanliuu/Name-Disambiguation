cd src
git add .
echo -n "Enter your commit explanation: "
read message
git commit -m "$message"
git pull origin master
