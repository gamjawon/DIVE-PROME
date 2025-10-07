// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'route_info_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
RouteInfoModel _$RouteInfoModelFromJson(
  Map<String, dynamic> json
) {
    return _RouteInfo.fromJson(
      json
    );
}

/// @nodoc
mixin _$RouteInfoModel {

@JsonKey(name: 'label') String get label;@JsonKey(name: 'path_points', fromJson: _convertPathPoints) List<Map<String, double>> get pathPoints;@JsonKey(name: 'distance_m') double get distanceM;@JsonKey(name: 'duration_sec') int get durationSec;@JsonKey(name: 'lane_changes') int get laneChanges;@JsonKey(name: 'u_turns') int get uTurns;@JsonKey(name: 'steep_slopes') int get steepSlopes;
/// Create a copy of RouteInfoModel
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$RouteInfoModelCopyWith<RouteInfoModel> get copyWith => _$RouteInfoModelCopyWithImpl<RouteInfoModel>(this as RouteInfoModel, _$identity);

  /// Serializes this RouteInfoModel to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is RouteInfoModel&&(identical(other.label, label) || other.label == label)&&const DeepCollectionEquality().equals(other.pathPoints, pathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,label,const DeepCollectionEquality().hash(pathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfoModel(label: $label, pathPoints: $pathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class $RouteInfoModelCopyWith<$Res>  {
  factory $RouteInfoModelCopyWith(RouteInfoModel value, $Res Function(RouteInfoModel) _then) = _$RouteInfoModelCopyWithImpl;
@useResult
$Res call({
@JsonKey(name: 'label') String label,@JsonKey(name: 'path_points', fromJson: _convertPathPoints) List<Map<String, double>> pathPoints,@JsonKey(name: 'distance_m') double distanceM,@JsonKey(name: 'duration_sec') int durationSec,@JsonKey(name: 'lane_changes') int laneChanges,@JsonKey(name: 'u_turns') int uTurns,@JsonKey(name: 'steep_slopes') int steepSlopes
});




}
/// @nodoc
class _$RouteInfoModelCopyWithImpl<$Res>
    implements $RouteInfoModelCopyWith<$Res> {
  _$RouteInfoModelCopyWithImpl(this._self, this._then);

  final RouteInfoModel _self;
  final $Res Function(RouteInfoModel) _then;

/// Create a copy of RouteInfoModel
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? label = null,Object? pathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_self.copyWith(
label: null == label ? _self.label : label // ignore: cast_nullable_to_non_nullable
as String,pathPoints: null == pathPoints ? _self.pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
as List<Map<String, double>>,distanceM: null == distanceM ? _self.distanceM : distanceM // ignore: cast_nullable_to_non_nullable
as double,durationSec: null == durationSec ? _self.durationSec : durationSec // ignore: cast_nullable_to_non_nullable
as int,laneChanges: null == laneChanges ? _self.laneChanges : laneChanges // ignore: cast_nullable_to_non_nullable
as int,uTurns: null == uTurns ? _self.uTurns : uTurns // ignore: cast_nullable_to_non_nullable
as int,steepSlopes: null == steepSlopes ? _self.steepSlopes : steepSlopes // ignore: cast_nullable_to_non_nullable
as int,
  ));
}

}


/// Adds pattern-matching-related methods to [RouteInfoModel].
extension RouteInfoModelPatterns on RouteInfoModel {
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function(@JsonKey(name: 'label')  String label, @JsonKey(name: 'path_points', fromJson: _convertPathPoints)  List<Map<String, double>> pathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.label,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function(@JsonKey(name: 'label')  String label, @JsonKey(name: 'path_points', fromJson: _convertPathPoints)  List<Map<String, double>> pathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)  $default,) {final _that = this;
switch (_that) {
case _RouteInfo():
return $default(_that.label,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function(@JsonKey(name: 'label')  String label, @JsonKey(name: 'path_points', fromJson: _convertPathPoints)  List<Map<String, double>> pathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)?  $default,) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.label,_that.pathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _RouteInfo implements RouteInfoModel {
  const _RouteInfo({@JsonKey(name: 'label') required this.label, @JsonKey(name: 'path_points', fromJson: _convertPathPoints) final  List<Map<String, double>> pathPoints = const [], @JsonKey(name: 'distance_m') this.distanceM = 0.0, @JsonKey(name: 'duration_sec') this.durationSec = 0, @JsonKey(name: 'lane_changes') this.laneChanges = 0, @JsonKey(name: 'u_turns') this.uTurns = 0, @JsonKey(name: 'steep_slopes') this.steepSlopes = 0}): _pathPoints = pathPoints;
  factory _RouteInfo.fromJson(Map<String, dynamic> json) => _$RouteInfoFromJson(json);

@override@JsonKey(name: 'label') final  String label;
 final  List<Map<String, double>> _pathPoints;
@override@JsonKey(name: 'path_points', fromJson: _convertPathPoints) List<Map<String, double>> get pathPoints {
  if (_pathPoints is EqualUnmodifiableListView) return _pathPoints;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_pathPoints);
}

@override@JsonKey(name: 'distance_m') final  double distanceM;
@override@JsonKey(name: 'duration_sec') final  int durationSec;
@override@JsonKey(name: 'lane_changes') final  int laneChanges;
@override@JsonKey(name: 'u_turns') final  int uTurns;
@override@JsonKey(name: 'steep_slopes') final  int steepSlopes;

/// Create a copy of RouteInfoModel
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$RouteInfoCopyWith<_RouteInfo> get copyWith => __$RouteInfoCopyWithImpl<_RouteInfo>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$RouteInfoToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _RouteInfo&&(identical(other.label, label) || other.label == label)&&const DeepCollectionEquality().equals(other._pathPoints, _pathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,label,const DeepCollectionEquality().hash(_pathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfoModel(label: $label, pathPoints: $pathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class _$RouteInfoCopyWith<$Res> implements $RouteInfoModelCopyWith<$Res> {
  factory _$RouteInfoCopyWith(_RouteInfo value, $Res Function(_RouteInfo) _then) = __$RouteInfoCopyWithImpl;
@override @useResult
$Res call({
@JsonKey(name: 'label') String label,@JsonKey(name: 'path_points', fromJson: _convertPathPoints) List<Map<String, double>> pathPoints,@JsonKey(name: 'distance_m') double distanceM,@JsonKey(name: 'duration_sec') int durationSec,@JsonKey(name: 'lane_changes') int laneChanges,@JsonKey(name: 'u_turns') int uTurns,@JsonKey(name: 'steep_slopes') int steepSlopes
});




}
/// @nodoc
class __$RouteInfoCopyWithImpl<$Res>
    implements _$RouteInfoCopyWith<$Res> {
  __$RouteInfoCopyWithImpl(this._self, this._then);

  final _RouteInfo _self;
  final $Res Function(_RouteInfo) _then;

/// Create a copy of RouteInfoModel
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? label = null,Object? pathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_RouteInfo(
label: null == label ? _self.label : label // ignore: cast_nullable_to_non_nullable
as String,pathPoints: null == pathPoints ? _self._pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
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
