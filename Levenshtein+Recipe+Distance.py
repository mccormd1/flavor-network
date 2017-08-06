
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# In[3]:


handle = open('srep00196-s3.csv')
head=5
count=0
ingredients=dict()
country=dict()
for recipe in handle:
    count+=1
#     print(count)
    if count < head:
        continue
    else:
        recipelist=recipe.strip().split(',')
        cname=recipelist[0]
        iname=recipelist
        country[cname]=country.get(cname,0)+1
        for item in iname:
            ingredients[item]=ingredients.get(item,0)+1
#         print(recipe.strip().split(','))
#         print(cname)
#         print(iname)
        if count%10000==0:
            print(count)
    


# In[4]:

country


# In[36]:

ingredients


# In[37]:

ingredlist=list(ingredients.keys())
ingredlist


# In[38]:

r=sum(country.values())
# ingredlist.insert(0,'country')
c=len(ingredlist)


# In[8]:

# reciperemap=pd.DataFrame(columns=ingredlist)


# In[41]:

ingredlist.index('African')


# In[75]:

handle = open('srep00196-s3.csv')
head=4
count=0
rlist=[]
for recipe in handle:
    itemlist=[0]*c
#     print(count)
    if count < head:
        count+=1
        continue
    else:
        recipelist=recipe.strip().split(',')
        cname=recipelist[0]
        iname=recipelist
#         reciperemap['country'].append(cname)
        for item in iname:
            itemlist[ingredlist.index(item)]=1
#         print(recipe.strip().split(','))
#         print(cname)
#         print(iname)
#         itemlist.insert(0,cname)
        rlist.append(itemlist)
        count+=1
        if count%10000==0:
            print(count)
#             break
    


# In[76]:

c


# In[77]:

len(rlist[0])


# In[78]:

# ingredlist.insert(0,'country')
reciperemap=pd.DataFrame(rlist,columns=ingredlist)


# In[79]:

reciperemap


# In[80]:

mapc=reciperemap.copy()
coocc=mapc.T.dot(mapc)
# np.fill_diagonal(coocc.values,0)


# In[81]:

reciperemap[reciperemap!=0].stack()


# In[469]:

reciperemap.query('African>0').values.sum()


# In[420]:

coocclog=coocc.replace(0,.00000001)
coocclog=np.log(coocclog)


# In[421]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots()
fig.set_size_inches(50,50)
sns.heatmap(coocclog)
plt.show()


# In[614]:

list(country.keys())


# In[617]:

fig, ax = plt.subplots()
fig.set_size_inches(50,50)
sns.heatmap(coocclog[list(country.keys())])
plt.show()


# In[86]:

coocc['garlic'].sort_values(ascending=False)


# In[103]:

len(coocc.query('African>0'))


# In[490]:

countrylist=list(country.keys())
variety=dict()
totalcuisineingredients=dict()
for cuisine in countrylist:
    queryname=cuisine+'>0'
    variety.update({cuisine:len(coocc.query(queryname))})
    totalcuisineingredients.update({cuisine:reciperemap.query(queryname).sum(axis=0).drop(cuisine).sum()})
variety=pd.DataFrame(variety,index=[0])
totalcuisineingredients=pd.DataFrame(totalcuisineingredients,index=[0])                             
variety


# In[491]:

totalcuisineingredients


# In[400]:

fig, ax = plt.subplots()
fig.set_size_inches(10,5)
plt.xticks(rotation=45)
ax=sns.barplot(data=variety)
ax.set(xlabel='Cuisine', ylabel='# Unique Ingredients')

plt.show()


# In[ ]:




# In[495]:

uniqueperrecipe=variety/list(country.values())
uniqueperrecipe


# In[594]:

avgnumingredients=totalcuisineingredients/list(country.values())
avgnumingredients


# In[595]:

sns.color_palette("husl", 11)


# In[609]:

fig, ax = plt.subplots(4,sharex=True)
cmap=sns.color_palette("husl", 11)
fig.set_size_inches(10,10)
plt.xticks(rotation=90)
N = 11
ind = np.arange(N)  # the x locations for the groups
width = 0.35
ax[0].bar(ind,variety.values.tolist()[0],color=cmap,tick_label=countrylist)
ax[0].set(ylabel='# Unique Ingredients')
ax[1].set(ylabel="# recipes")
ax[1].bar(ind,list(country.values()),color=cmap,tick_label=countrylist)
ax[2].set(ylabel="Avg Unique per recipe")
ax[2].bar(ind,uniqueperrecipe.values.tolist()[0],color=cmap,tick_label=countrylist)
ax[3].set(xlabel='Cuisine', ylabel="Avg ingredients per recipe")
ax[3].bar(ind,avgnumingredients.values.tolist()[0],color=cmap,tick_label=countrylist)

plt.show()


# In[139]:

orderedrecipes=reciperemap[reciperemap!=0].stack()


# In[492]:




# In[412]:

def closestfit(recipe,cuisine,forceingredient=0):
    if forceingredient == 0: #picks recipe with fewest number of additional items, even if no ingredient overlap
#         reciperemap.loc[recipe]
        queryname=cuisine+'==1'
        booleanmat=reciperemap.query(queryname)==reciperemap.loc[0]
        sumbool=booleanmat.sum(1)
        closestrecipe=sumbool[sumbool==max(sumbool)]
        closestrecipeingredients=reciperemap.loc[closestrecipe.keys()[0]]
    elif forceingredient == 1: #forces to pick recipes with at least one shared ingredient
        closestrecipeingredients=None
    return closestrecipeingredients.name


# In[413]:

closestfit(0,'EastAsian')


# In[414]:

startrecipe=0
cuisine='EastAsian'
endrecipe=closestfit(startrecipe,cuisine)
endrecipe


# In[265]:

er=reciperemap.loc[endrecipe][reciperemap.loc[endrecipe]==1]
sr=reciperemap.loc[startrecipe][reciperemap.loc[startrecipe]==1]
srlist=list(sr.keys().drop(countrylist,errors='ignore'))
erlist=list(er.keys().drop(countrylist,errors='ignore'))


# In[339]:

srlist


# In[340]:

srlist.pop(4)
srlist


# In[341]:

# levenshteindist
erlist


# In[342]:

def ingredientreorder(srlist,erlist):
    if len(erlist) >= len(srlist):
        big=erlist
        small=srlist
        returntype=0
    else:
        big=srlist
        small=erlist
        returntype=1
        
    for i,item in enumerate(small):
#         print(item)
        try:
            loc=big.index(item)
    #         print(loc)
            big.pop(loc)
#             print(i,loc,item)
            big.insert(i,item)
        except:
            continue
    if returntype == 0:
        return small, big
    elif returntype == 1:
        return big, small
    


# In[343]:

start,end=ingredientreorder(srlist,erlist)


# In[344]:

start


# In[345]:

end


# In[346]:

def iterative_levenshtein(s, t):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    for r in range(rows):
        print(dist[r])
    
 
    return dist[row][col]


# In[347]:

iterative_levenshtein(start,end)


# In[ ]:




# In[415]:

# ingredientrecommender(10,'SouthAsian')
startrecipe=3660
recipe=reciperemap.loc[startrecipe]
cuisine='African'
endrecipe=closestfit(recipe,cuisine)
er=reciperemap.loc[endrecipe][reciperemap.loc[endrecipe]==1]
sr=reciperemap.loc[startrecipe][reciperemap.loc[startrecipe]==1]
srlist=list(sr.keys().drop(countrylist,errors='ignore'))
erlist=list(er.keys().drop(countrylist,errors='ignore'))
print(sr,er)
# start,end=ingredientreorder(srlist,erlist)
# start


# In[392]:

# coocc['cardamom'].sort_values(ascending=False)


# In[ ]:




# In[ ]:



