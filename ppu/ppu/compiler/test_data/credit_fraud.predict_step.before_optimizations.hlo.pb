
�2
lcluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_1_.73lcluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_1_.73�
%binary_crossentropy_Mean-reduction.29!
x.30	parameter* : �� #
y.31	parameter* : H�� "
add.32add* : � �� "$
* 
* * x.30y.31(0 �
2binary_crossentropy_weighted_loss_Sum-reduction.42!
x.43	parameter* : �+� #
y.44	parameter* : H�,� "
add.45add* : �-�+,� "$
* 
* * x.43y.44(*0-�-
lcluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_6__XlaNumResourceArgs_1_.73c
constant.60constant*
  2 :
StridedSlicestrided_sliceB
*
  2 " �<� c
constant.61constant*
  2 :
StridedSlicestrided_sliceB
*
  2 "�=� c
constant.62constant*
  2 :
StridedSlicestrided_sliceB
*
  2 "�>� A

constant.7constant* :
ShapeShapeB
* "d�� 6
	convert.8convert* :
ShapeShape��� C
broadcast.9	broadcast*
  2 :
ShapeShape�	�� B
constant.10constant* :
ShapeShapeB
* "�
� 7

convert.11convert* :
ShapeShape��
� D
broadcast.12	broadcast*
  2 :
ShapeShape��� L
concatenate.13concatenate*
  2 :
ShapeShaper ��	� R
slice.63slice*
  2 :
StridedSlicestrided_slice��?�� F

reshape.64reshape* :
StridedSlicestrided_slice�@�?� 5

convert.65convert* :
CastCast�A�@� 6

reshape.68reshape* :XLA_Retvals�D�A� 5
arg2.3	parameter* :
XLA_ArgsH��
 � j
constant.48constant* :6
Size.binary_crossentropy/weighted_loss/num_elementsB
* 2�0� j
constant.49constant* :6
Size.binary_crossentropy/weighted_loss/num_elementsB
* "d�1� _

convert.50convert* :6
Size.binary_crossentropy/weighted_loss/num_elements�2�1� b
multiply.51multiply* :6
Size.binary_crossentropy/weighted_loss/num_elements�3�02� _

convert.52convert* :6
Size.binary_crossentropy/weighted_loss/num_elements�4�3� d

convert.53convert* :;
Cast3binary_crossentropy/weighted_loss/num_elements/Cast�5�4� j
constant.54constant* :3
DivNoNan'binary_crossentropy/weighted_loss/valueB
* B    �6� j

compare.55compare* :3
DivNoNan'binary_crossentropy/weighted_loss/value�7�56�EQ� �FLOATj
constant.56constant* :3
DivNoNan'binary_crossentropy/weighted_loss/valueB
* B    �8� `
broadcast.57	broadcast* :3
DivNoNan'binary_crossentropy/weighted_loss/value�9�8� A
arg1.2	parameterd*
  2  :
XLA_ArgsH��
 � 4
	reshape.5reshaped*
  2  : ��� p
constant.17constant* :9
	ZerosLike,binary_crossentropy/logistic_loss/zeros_likeB
* B    �� r
broadcast.18	broadcastd*
  2  :9
	ZerosLike,binary_crossentropy/logistic_loss/zeros_like��� �

compare.19compared*
  2  :>
GreaterEqual.binary_crossentropy/logistic_loss/GreaterEqual���GE� �FLOAT_
	negate.15negated*
  2  :,
Neg%binary_crossentropy/logistic_loss/Neg��� a
	select.20selectd*
  2  :,*binary_crossentropy/logistic_loss/Select_1��� �
exponential.21exponentiald*
  2  :U
_UnaryOpsComposition=binary_crossentropy/logistic_loss/Log1p/unary_ops_composition��� �
log-plus-one.22log-plus-oned*
  2  :U
_UnaryOpsComposition=binary_crossentropy/logistic_loss/Log1p/unary_ops_composition��� _
	select.23selectd*
  2  :*(binary_crossentropy/logistic_loss/Select��� <
arg0.1	parameterd*
  2 :
XLA_Args��
 � 1
	reshape.4reshaped*
  2 : ��� L
	reshape.6reshaped*
  2  :

ExpandDims
ExpandDims��� U

convert.14convertd*
  2  : 
Castbinary_crossentropy/Cast��� d
multiply.16multiplyd*
  2  :,
Mul%binary_crossentropy/logistic_loss/mul��� d
subtract.24subtractd*
  2  :,
Sub%binary_crossentropy/logistic_loss/sub��� X
add.25addd*
  2  :*
AddV2!binary_crossentropy/logistic_loss��� U

convert.26convertd*
  2  : 
Meanbinary_crossentropy/Mean��� W
constant.27constant* : 
Meanbinary_crossentropy/MeanB
* B    �� I

convert.28convert* : 
Meanbinary_crossentropy/Mean��� X
	reduce.33reduced*
  2 : 
Meanbinary_crossentropy/Meanr�!��� T
constant.34constant* : 
Meanbinary_crossentropy/MeanB
* "�"� I

convert.35convert* : 
Meanbinary_crossentropy/Mean�#�"� V
broadcast.36	broadcastd*
  2 : 
Meanbinary_crossentropy/Mean�$�#� Q
	divide.37divided*
  2 : 
Meanbinary_crossentropy/Mean�%�!$� R

convert.38convertd*
  2 : 
Meanbinary_crossentropy/Mean�&�%� ^

convert.39convertd*
  2 :,
Sum%binary_crossentropy/weighted_loss/Sum�'�&� c
constant.40constant* :,
Sum%binary_crossentropy/weighted_loss/SumB
* B    �(� U

convert.41convert* :,
Sum%binary_crossentropy/weighted_loss/Sum�)�(� [
	reduce.46reduce* :,
Sum%binary_crossentropy/weighted_loss/Sumr �.�')�*� U

convert.47convert* :,
Sum%binary_crossentropy/weighted_loss/Sum�/�.� [
	divide.58divide* :3
DivNoNan'binary_crossentropy/weighted_loss/value�:�/5� \
	select.59select* :3
DivNoNan'binary_crossentropy/weighted_loss/value�;�79:� 6
multiply.66multiply* :

MulMul�B�A;� L
add.67add* :*
AssignAddVariableOpAssignAddVariableOp�C�B� 6

reshape.69reshape* :XLA_Retvals�E�C� 6
tuple.70tuple
"* :XLA_Retvals�F�E� J
get-tuple-element.71get-tuple-element* :XLA_Retvals�G�F� ?
tuple.72tuple"* "* :XLA_Retvals�H�DG� "Y
d*
  2 
d*
  2  
* "* "* arg0.1arg1.2arg2.3(I0H"M
d*
  2 
d*
  2  
* "* "* p0p1p2(0IB J 