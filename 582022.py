#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# dataframe = veri çerçevesi

# In[4]:


from numpy.random import randn


# In[7]:


seri1=np.random.randn(10)


# In[8]:


seri1


# In[9]:


randn(3)


# In[10]:


np.random.randn(10)


# In[11]:


randn(3,3)


# In[17]:


df1=pd.DataFrame(data=randn(3,3),index=["A1","B1","C1"],columns=["S1","S2","S3"])


# In[18]:


df1


# In[19]:


type(df1)


# In[20]:


df1["S1"]


# In[22]:


df1["S2"]


# In[24]:


a=df1[["S1","S2"]]


# In[25]:


df1["S4"]=pd.Series(randn(3),["A1","B1","C1"])


# In[26]:


df1


# In[27]:


df1.drop("S2",axis=1,inplace=True)


# In[28]:


df1


# # Statsmodels

# In[31]:


import statsmodels.api as sm


# In[32]:


veri=pd.read_excel("c:\python1\data.xlsx")


# In[33]:


print(veri)


# In[34]:


y=veri["TANSIYON"]


# In[59]:


x=veri["KILO"]


# In[60]:


x=sm.add_constant(x)


# In[78]:


çıktı=sm.OLS(y,x).fit()


# In[79]:


çıktı.summary()


# KILO daki 1 birimlik artış, TANSIYON değişkenin ortalama 0.7638 birimlik artış yaratır.

# TANSIYON daki değişimin ortalama %94.2 lik kısmını KILO'daki değişim açıklamaktadır.

# F istatistiği modelin anlamlılığını test etmeye yarar; diğer bir ifadeyle R2'nin anlamlılığını sınamaya yarar, diğer bir ifadeyle sabit terim dışındaki katsayıların anlamlılığını sınamaya yarar.

# In[80]:


12.090**2


# In[81]:


çıktı.rsquared


# In[82]:


çıktı.rsquared_adj


# In[83]:


çıktı.fvalue


# In[84]:


a=çıktı.fvalue


# In[85]:


çıktı.params


# In[86]:


çıktı.params.KILO


# In[87]:


çıktı.params.const


# In[88]:


çıktı.params[0]


# In[89]:


çıktı.params[1]


# In[91]:


print("Kilo değişkenindeki 1 birimlik artış, Tansiyon değişkeninde",çıktı.params[1],"birimlik artış yaratır.")


# In[92]:


çıktı.tvalues


# In[93]:


çıktı.tvalues.const


# In[94]:


çıktı.tvalues.KILO


# In[95]:


çıktı.tvalues[0]


# In[96]:


çıktı.tvalues[1]


# In[97]:


çıktı.pvalues


# In[98]:


çıktı.pvalues.KILO


# In[99]:


çıktı.pvalues.const


# In[100]:


çıktı.pvalues[0]


# In[101]:


çıktı.pvalues[1]


# ### Çoklu Regresyon Modeli

# In[102]:


w=veri["YAS"]


# In[103]:


print(w)


# In[104]:


model=sm.OLS.from.formula("y~x+w")


# In[105]:


print(veri)


# In[119]:


model=sm.OLS.from_formula("TANSIYON~KILO+YAS", veri)


# In[120]:


çıktı=model.fit()


# In[121]:


çıktı.summary()


# ## Varsayımların Sınanması

# Breusch-Godfrey otokorelasyon testi bir ve daha yüksek dereceli otokorelasyonun varlığını sınamak için kullanılır. Temel hipotez modelde ilgili derecede otokorelasyon olmadığını göstermektedir.

# In[122]:


import statsmodels.stats.diagnostic as dg


# In[124]:


print(dg.acorr_breusch_godfrey(çıktı,nlags=1))


# In[125]:


print(dg.acorr_breusch_godfrey(çıktı,nlags=2))


# Breusch-Pagan ve White testleri modelde sabit varyans varsayımının geçerliliğini sınamak için kullanılırlar. Teml hipotezde modelde sabit varyans varsayımının geçerli olduğu yer alır.

# In[127]:


çıktı.resid


# In[128]:


print(dg.het_breuschpagan(çıktı.resid,çıktı.model.exog))


# ut^2=yaş~kilo

# In[129]:


print(dg.het_white(çıktı.resid,çıktı.model.exog))


# ut^2=yas, kilo, yas^2, kilo^2, yaş*kilo

# DW testinin uygulanması için dL ve dU kritik değerlerine ihtiyaç duyulur.

# ##  Zaman Serileri

# In[130]:


iss=pd.read_excel("c:\python1\issizlik.xlsx")


# In[131]:


print(iss)


# In[132]:


from matplotlib import pyplot


# In[134]:


iss.plot()
pyplot.show()


# DY=b0+b1Y(t-1)+b2DY(t-1)+b3DY(t-2)....+b11DY(t-10)

# ADF birim kök testinde temel hipotez altında ilgilenilen serinin birim köklü olduğu varsayım yer alır.

# AIC, BIC

# t-stat

# In[135]:


import statsmodels.tsa.stattools as ts


# In[136]:


print(ts.adfuller(iss))


# maxgecikme=12*(T/100)^0.25

# In[141]:


ts.adfuller(iss,maxlag=4,regression="c",autolag="t-stat", store=True, regresults=True)


# KPSS; temel hipotez altında durağanlk sınanır.

# In[143]:


ts.kpss(iss,regression="c",nlags="auto",store=True)


# Zivot-Andrews

# D
# 0
# 0
# 0
# 0
# 0
# 0
# 0
# 1
# 1
# 1
# 1
# 

# DY=b0+b1Y(t-1)+DU
# 

# kırpma bölgesi = %15
# 

# t1, t2,...,t70

# In[146]:


print(ts.zivot_andrews(iss,trim=0.10,maxlag=3,regression="c",autolag="t-stat"))


# ## Engle-Granger eşbütünleşme Testi

# In[149]:


print(y)


# In[151]:


print(w)


# In[152]:


ts.coint(y,w)


# In[153]:


print(ts.coint(y,w))


# In[154]:


ts.coint(y,w,trend="c",method="aeg",maxlag=4,autolag="t-stat")


# ## Granger Nedensellik Testi

# In[155]:


veri


# In[160]:


veri.drop("TANSIYON",axis=1,inplace=True)


# In[161]:


print(veri)


# In[164]:


print(ts.grangercausalitytests(veri,maxlag=2,addconst=True))


# In[ ]:




