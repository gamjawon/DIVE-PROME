// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'route_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$RouteInfo {

 String get label;@JsonKey(name: 'path_points') List<List<double>> get pathPoints;@JsonKey(name: 'display_path_points') List<List<double>> get displayPathPoints;@JsonKey(name: 'distance_m') double get distanceM;@JsonKey(name: 'duration_sec') int get durationSec;@JsonKey(name: 'lane_changes') int get laneChanges;@JsonKey(name: 'u_turns') int get uTurns;@JsonKey(name: 'steep_slopes') int get steepSlopes;
/// Create a copy of RouteInfo
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$RouteInfoCopyWith<RouteInfo> get copyWith => _$RouteInfoCopyWithImpl<RouteInfo>(this as RouteInfo, _$identity);

  /// Serializes this RouteInfo to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is RouteInfo&&(identical(other.label, label) || other.label == label)&&const DeepCollectionEquality().equals(other.pathPoints, pathPoints)&&const DeepCollectionEquality().equals(other.displayPathPoints, displayPathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,label,const DeepCollectionEquality().hash(pathPoints),const DeepCollectionEquality().hash(displayPathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfo(label: $label, pathPoints: $pathPoints, displayPathPoints: $displayPathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class $RouteInfoCopyWith<$Res>  {
  factory $RouteInfoCopyWith(RouteInfo value, $Res Function(RouteInfo) _then) = _$RouteInfoCopyWithImpl;
@useResult
$Res call({
 String label,@JsonKey(name: 'path_points') List<List<double>> pathPoints,@JsonKey(name: 'display_path_points') List<List<double>> displayPathPoints,@JsonKey(name: 'distance_m') double distanceM,@JsonKey(name: 'duration_sec') int durationSec,@JsonKey(name: 'lane_changes') int laneChanges,@JsonKey(name: 'u_turns') int uTurns,@JsonKey(name: 'steep_slopes') int steepSlopes
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
@pragma('vm:prefer-inline') @override $Res call({Object? label = null,Object? pathPoints = null,Object? displayPathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_self.copyWith(
label: null == label ? _self.label : label // ignore: cast_nullable_to_non_nullable
as String,pathPoints: null == pathPoints ? _self.pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
as List<List<double>>,displayPathPoints: null == displayPathPoints ? _self.displayPathPoints : displayPathPoints // ignore: cast_nullable_to_non_nullable
as List<List<double>>,distanceM: null == distanceM ? _self.distanceM : distanceM // ignore: cast_nullable_to_non_nullable
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( String label, @JsonKey(name: 'path_points')  List<List<double>> pathPoints, @JsonKey(name: 'display_path_points')  List<List<double>> displayPathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.label,_that.pathPoints,_that.displayPathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( String label, @JsonKey(name: 'path_points')  List<List<double>> pathPoints, @JsonKey(name: 'display_path_points')  List<List<double>> displayPathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)  $default,) {final _that = this;
switch (_that) {
case _RouteInfo():
return $default(_that.label,_that.pathPoints,_that.displayPathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( String label, @JsonKey(name: 'path_points')  List<List<double>> pathPoints, @JsonKey(name: 'display_path_points')  List<List<double>> displayPathPoints, @JsonKey(name: 'distance_m')  double distanceM, @JsonKey(name: 'duration_sec')  int durationSec, @JsonKey(name: 'lane_changes')  int laneChanges, @JsonKey(name: 'u_turns')  int uTurns, @JsonKey(name: 'steep_slopes')  int steepSlopes)?  $default,) {final _that = this;
switch (_that) {
case _RouteInfo() when $default != null:
return $default(_that.label,_that.pathPoints,_that.displayPathPoints,_that.distanceM,_that.durationSec,_that.laneChanges,_that.uTurns,_that.steepSlopes);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _RouteInfo extends RouteInfo {
  const _RouteInfo({required this.label, @JsonKey(name: 'path_points') final  List<List<double>> pathPoints = const [], @JsonKey(name: 'display_path_points') final  List<List<double>> displayPathPoints = const [], @JsonKey(name: 'distance_m') this.distanceM = 0.0, @JsonKey(name: 'duration_sec') this.durationSec = 0, @JsonKey(name: 'lane_changes') this.laneChanges = 0, @JsonKey(name: 'u_turns') this.uTurns = 0, @JsonKey(name: 'steep_slopes') this.steepSlopes = 0}): _pathPoints = pathPoints,_displayPathPoints = displayPathPoints,super._();
  factory _RouteInfo.fromJson(Map<String, dynamic> json) => _$RouteInfoFromJson(json);

@override final  String label;
 final  List<List<double>> _pathPoints;
@override@JsonKey(name: 'path_points') List<List<double>> get pathPoints {
  if (_pathPoints is EqualUnmodifiableListView) return _pathPoints;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_pathPoints);
}

 final  List<List<double>> _displayPathPoints;
@override@JsonKey(name: 'display_path_points') List<List<double>> get displayPathPoints {
  if (_displayPathPoints is EqualUnmodifiableListView) return _displayPathPoints;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_displayPathPoints);
}

@override@JsonKey(name: 'distance_m') final  double distanceM;
@override@JsonKey(name: 'duration_sec') final  int durationSec;
@override@JsonKey(name: 'lane_changes') final  int laneChanges;
@override@JsonKey(name: 'u_turns') final  int uTurns;
@override@JsonKey(name: 'steep_slopes') final  int steepSlopes;

/// Create a copy of RouteInfo
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
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _RouteInfo&&(identical(other.label, label) || other.label == label)&&const DeepCollectionEquality().equals(other._pathPoints, _pathPoints)&&const DeepCollectionEquality().equals(other._displayPathPoints, _displayPathPoints)&&(identical(other.distanceM, distanceM) || other.distanceM == distanceM)&&(identical(other.durationSec, durationSec) || other.durationSec == durationSec)&&(identical(other.laneChanges, laneChanges) || other.laneChanges == laneChanges)&&(identical(other.uTurns, uTurns) || other.uTurns == uTurns)&&(identical(other.steepSlopes, steepSlopes) || other.steepSlopes == steepSlopes));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,label,const DeepCollectionEquality().hash(_pathPoints),const DeepCollectionEquality().hash(_displayPathPoints),distanceM,durationSec,laneChanges,uTurns,steepSlopes);

@override
String toString() {
  return 'RouteInfo(label: $label, pathPoints: $pathPoints, displayPathPoints: $displayPathPoints, distanceM: $distanceM, durationSec: $durationSec, laneChanges: $laneChanges, uTurns: $uTurns, steepSlopes: $steepSlopes)';
}


}

/// @nodoc
abstract mixin class _$RouteInfoCopyWith<$Res> implements $RouteInfoCopyWith<$Res> {
  factory _$RouteInfoCopyWith(_RouteInfo value, $Res Function(_RouteInfo) _then) = __$RouteInfoCopyWithImpl;
@override @useResult
$Res call({
 String label,@JsonKey(name: 'path_points') List<List<double>> pathPoints,@JsonKey(name: 'display_path_points') List<List<double>> displayPathPoints,@JsonKey(name: 'distance_m') double distanceM,@JsonKey(name: 'duration_sec') int durationSec,@JsonKey(name: 'lane_changes') int laneChanges,@JsonKey(name: 'u_turns') int uTurns,@JsonKey(name: 'steep_slopes') int steepSlopes
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
@override @pragma('vm:prefer-inline') $Res call({Object? label = null,Object? pathPoints = null,Object? displayPathPoints = null,Object? distanceM = null,Object? durationSec = null,Object? laneChanges = null,Object? uTurns = null,Object? steepSlopes = null,}) {
  return _then(_RouteInfo(
label: null == label ? _self.label : label // ignore: cast_nullable_to_non_nullable
as String,pathPoints: null == pathPoints ? _self._pathPoints : pathPoints // ignore: cast_nullable_to_non_nullable
as List<List<double>>,displayPathPoints: null == displayPathPoints ? _self._displayPathPoints : displayPathPoints // ignore: cast_nullable_to_non_nullable
as List<List<double>>,distanceM: null == distanceM ? _self.distanceM : distanceM // ignore: cast_nullable_to_non_nullable
as double,durationSec: null == durationSec ? _self.durationSec : durationSec // ignore: cast_nullable_to_non_nullable
as int,laneChanges: null == laneChanges ? _self.laneChanges : laneChanges // ignore: cast_nullable_to_non_nullable
as int,uTurns: null == uTurns ? _self.uTurns : uTurns // ignore: cast_nullable_to_non_nullable
as int,steepSlopes: null == steepSlopes ? _self.steepSlopes : steepSlopes // ignore: cast_nullable_to_non_nullable
as int,
  ));
}


}


/// @nodoc
mixin _$RouteRequest {

 double get startLat; double get startLng; double get endLat; double get endLng;
/// Create a copy of RouteRequest
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$RouteRequestCopyWith<RouteRequest> get copyWith => _$RouteRequestCopyWithImpl<RouteRequest>(this as RouteRequest, _$identity);

  /// Serializes this RouteRequest to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is RouteRequest&&(identical(other.startLat, startLat) || other.startLat == startLat)&&(identical(other.startLng, startLng) || other.startLng == startLng)&&(identical(other.endLat, endLat) || other.endLat == endLat)&&(identical(other.endLng, endLng) || other.endLng == endLng));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,startLat,startLng,endLat,endLng);

@override
String toString() {
  return 'RouteRequest(startLat: $startLat, startLng: $startLng, endLat: $endLat, endLng: $endLng)';
}


}

/// @nodoc
abstract mixin class $RouteRequestCopyWith<$Res>  {
  factory $RouteRequestCopyWith(RouteRequest value, $Res Function(RouteRequest) _then) = _$RouteRequestCopyWithImpl;
@useResult
$Res call({
 double startLat, double startLng, double endLat, double endLng
});




}
/// @nodoc
class _$RouteRequestCopyWithImpl<$Res>
    implements $RouteRequestCopyWith<$Res> {
  _$RouteRequestCopyWithImpl(this._self, this._then);

  final RouteRequest _self;
  final $Res Function(RouteRequest) _then;

/// Create a copy of RouteRequest
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? startLat = null,Object? startLng = null,Object? endLat = null,Object? endLng = null,}) {
  return _then(_self.copyWith(
startLat: null == startLat ? _self.startLat : startLat // ignore: cast_nullable_to_non_nullable
as double,startLng: null == startLng ? _self.startLng : startLng // ignore: cast_nullable_to_non_nullable
as double,endLat: null == endLat ? _self.endLat : endLat // ignore: cast_nullable_to_non_nullable
as double,endLng: null == endLng ? _self.endLng : endLng // ignore: cast_nullable_to_non_nullable
as double,
  ));
}

}


/// Adds pattern-matching-related methods to [RouteRequest].
extension RouteRequestPatterns on RouteRequest {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _RouteRequest value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _RouteRequest() when $default != null:
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

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _RouteRequest value)  $default,){
final _that = this;
switch (_that) {
case _RouteRequest():
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _RouteRequest value)?  $default,){
final _that = this;
switch (_that) {
case _RouteRequest() when $default != null:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( double startLat,  double startLng,  double endLat,  double endLng)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _RouteRequest() when $default != null:
return $default(_that.startLat,_that.startLng,_that.endLat,_that.endLng);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( double startLat,  double startLng,  double endLat,  double endLng)  $default,) {final _that = this;
switch (_that) {
case _RouteRequest():
return $default(_that.startLat,_that.startLng,_that.endLat,_that.endLng);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( double startLat,  double startLng,  double endLat,  double endLng)?  $default,) {final _that = this;
switch (_that) {
case _RouteRequest() when $default != null:
return $default(_that.startLat,_that.startLng,_that.endLat,_that.endLng);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _RouteRequest extends RouteRequest {
  const _RouteRequest({required this.startLat, required this.startLng, required this.endLat, required this.endLng}): super._();
  factory _RouteRequest.fromJson(Map<String, dynamic> json) => _$RouteRequestFromJson(json);

@override final  double startLat;
@override final  double startLng;
@override final  double endLat;
@override final  double endLng;

/// Create a copy of RouteRequest
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$RouteRequestCopyWith<_RouteRequest> get copyWith => __$RouteRequestCopyWithImpl<_RouteRequest>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$RouteRequestToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _RouteRequest&&(identical(other.startLat, startLat) || other.startLat == startLat)&&(identical(other.startLng, startLng) || other.startLng == startLng)&&(identical(other.endLat, endLat) || other.endLat == endLat)&&(identical(other.endLng, endLng) || other.endLng == endLng));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,startLat,startLng,endLat,endLng);

@override
String toString() {
  return 'RouteRequest(startLat: $startLat, startLng: $startLng, endLat: $endLat, endLng: $endLng)';
}


}

/// @nodoc
abstract mixin class _$RouteRequestCopyWith<$Res> implements $RouteRequestCopyWith<$Res> {
  factory _$RouteRequestCopyWith(_RouteRequest value, $Res Function(_RouteRequest) _then) = __$RouteRequestCopyWithImpl;
@override @useResult
$Res call({
 double startLat, double startLng, double endLat, double endLng
});




}
/// @nodoc
class __$RouteRequestCopyWithImpl<$Res>
    implements _$RouteRequestCopyWith<$Res> {
  __$RouteRequestCopyWithImpl(this._self, this._then);

  final _RouteRequest _self;
  final $Res Function(_RouteRequest) _then;

/// Create a copy of RouteRequest
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? startLat = null,Object? startLng = null,Object? endLat = null,Object? endLng = null,}) {
  return _then(_RouteRequest(
startLat: null == startLat ? _self.startLat : startLat // ignore: cast_nullable_to_non_nullable
as double,startLng: null == startLng ? _self.startLng : startLng // ignore: cast_nullable_to_non_nullable
as double,endLat: null == endLat ? _self.endLat : endLat // ignore: cast_nullable_to_non_nullable
as double,endLng: null == endLng ? _self.endLng : endLng // ignore: cast_nullable_to_non_nullable
as double,
  ));
}


}

// dart format on
