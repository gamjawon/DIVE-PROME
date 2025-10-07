// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'route_info.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$RouteInfo {

 RouteOption get option; List<Map<String, double>> get pathPoints; double get distanceM; int get durationSec; int get laneChanges; int get uTurns; int get steepSlopes;
/// Create a copy of RouteInfo
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$RouteInfoCopyWith<RouteInfo> get copyWith => _$RouteInfoCopyWithImpl<RouteInfo>(this as RouteInfo, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is RouteInfo&&(identical(other.option, option) || other.option == option)&&const DeepCollectionEquality().equals(other.pathPoints, pathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}


@override
int get hashCode => Object.hash(runtimeType,option,const DeepCollectionEquality().hash(pathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfo(option: $option, pathPoints: $pathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class $RouteInfoCopyWith<$Res>  {
  factory $RouteInfoCopyWith(RouteInfo value, $Res Function(RouteInfo) _then) = _$RouteInfoCopyWithImpl;
@useResult
$Res call({
 RouteOption option, List<Map<String, double>> pathPoints, double distanceM, int durationSec, int laneChanges, int uTurns, int steepSlopes
});




}
/// @nodoc
class _$RouteInfoCopyWithImpl<$Res>
    implements $RouteInfoCopyWith<$Res> {
  _$RouteInfoCopyWithImpl(this._self, this._then);

  final RouteInfo _self;
  final $Res Function(RouteInfo) _then;

/// Create a copy of RouteInfo
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? option = null,Object? pathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_self.copyWith(
option: null == option ? _self.option : option // ignore: cast_nullable_to_non_nullable
as RouteOption,pathPoints: null == pathPoints ? _self.pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
as List<Map<String, double>>,distanceM: null == distanceM ? _self.distanceM : distanceM // ignore: cast_nullable_to_non_nullable
as double,durationSec: null == durationSec ? _self.durationSec : durationSec // ignore: cast_nullable_to_non_nullable
as int,laneChanges: null == laneChanges ? _self.laneChanges : laneChanges // ignore: cast_nullable_to_non_nullable
as int,uTurns: null == uTurns ? _self.uTurns : uTurns // ignore: cast_nullable_to_non_nullable
as int,steepSlopes: null == steepSlopes ? _self.steepSlopes : steepSlopes // ignore: cast_nullable_to_non_nullable
as int,
  ));
}

}


/// Adds pattern-matching-related methods to [RouteInfo].
extension RouteInfoPatterns on RouteInfo {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _RouteInfo value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _RouteInfo value)  $default,){
final _that = this;
switch (_that) {
case _RouteInfo():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _RouteInfo value)?  $default,){
final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( RouteOption option,  List<Map<String, double>> pathPoints,  double distanceM,  int durationSec,  int laneChanges,  int uTurns,  int steepSlopes)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.option,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( RouteOption option,  List<Map<String, double>> pathPoints,  double distanceM,  int durationSec,  int laneChanges,  int uTurns,  int steepSlopes)  $default,) {final _that = this;
switch (_that) {
case _RouteInfo():
return $default(_that.option,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( RouteOption option,  List<Map<String, double>> pathPoints,  double distanceM,  int durationSec,  int laneChanges,  int uTurns,  int steepSlopes)?  $default,) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.option,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
  return null;

}
}

}

/// @nodoc


class _RouteInfo extends RouteInfo {
  const _RouteInfo({required this.option, required final  List<Map<String, double>> pathPoints, required this.distanceM, required this.durationSec, required this.laneChanges, required this.uTurns, required this.steepSlopes}): _pathPoints = pathPoints,super._();
  

@override final  RouteOption option;
 final  List<Map<String, double>> _pathPoints;
@override List<Map<String, double>> get pathPoints {
  if (_pathPoints is EqualUnmodifiableListView) return _pathPoints;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_pathPoints);
}

@override final  double distanceM;
@override final  int durationSec;
@override final  int laneChanges;
@override final  int uTurns;
@override final  int steepSlopes;

/// Create a copy of RouteInfo
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$RouteInfoCopyWith<_RouteInfo> get copyWith => __$RouteInfoCopyWithImpl<_RouteInfo>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _RouteInfo&&(identical(other.option, option) || other.option == option)&&const DeepCollectionEquality().equals(other._pathPoints, _pathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}


@override
int get hashCode => Object.hash(runtimeType,option,const DeepCollectionEquality().hash(_pathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfo(option: $option, pathPoints: $pathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class _$RouteInfoCopyWith<$Res> implements $RouteInfoCopyWith<$Res> {
  factory _$RouteInfoCopyWith(_RouteInfo value, $Res Function(_RouteInfo) _then) = __$RouteInfoCopyWithImpl;
@override @useResult
$Res call({
 RouteOption option, List<Map<String, double>> pathPoints, double distanceM, int durationSec, int laneChanges, int uTurns, int steepSlopes
});




}
/// @nodoc
class __$RouteInfoCopyWithImpl<$Res>
    implements _$RouteInfoCopyWith<$Res> {
  __$RouteInfoCopyWithImpl(this._self, this._then);

  final _RouteInfo _self;
  final $Res Function(_RouteInfo) _then;

/// Create a copy of RouteInfo
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? option = null,Object? pathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_RouteInfo(
option: null == option ? _self.option : option // ignore: cast_nullable_to_non_nullable
as RouteOption,pathPoints: null == pathPoints ? _self._pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
as List<Map<String, double>>,distanceM: null == distanceM ? _self.distanceM : distanceM // ignore: cast_nullable_to_non_nullable
as double,durationSec: null == durationSec ? _self.durationSec : durationSec // ignore: cast_nullable_to_non_nullable
as int,laneChanges: null == laneChanges ? _self.laneChanges : laneChanges // ignore: cast_nullable_to_non_nullable
as int,uTurns: null == uTurns ? _self.uTurns : uTurns // ignore: cast_nullable_to_non_nullable
as int,steepSlopes: null == steepSlopes ? _self.steepSlopes : steepSlopes // ignore: cast_nullable_to_non_nullable
as int,
  ));
}


}

// dart format on
