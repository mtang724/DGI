## declare an array variable
# declare -a arr=("element1" "element2" "element3")

## now loop through the above array
for i in "cora" "citeseer" "pubmed" "chameleon" "squirrel" "film" "texas" "wisconsin"
do
   echo python test.py "$i"
   # or do whatever with individual element of the array
done