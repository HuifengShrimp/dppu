# Semi2k
| kernel |     latency     |            comm            |
|--------|-----------------|----------------------------|
|A2B     |(log(k)+1)*log(n)|(2*log(k)+1)*2*k*(n-1)*(n-1)|
|B2A     |1                |k*(n-1)                     |
|A2P     |1                |k                           |
|B2P     |1                |k                           |
|AddBB   |log(k)+1         |log(k)*k                    |
|MatMulAA|1                |Unknown                     |
|MatMulAP|0                |0                           |
