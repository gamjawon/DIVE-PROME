// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'location_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
LocationModel _$LocationModelFromJson(
  Map<String, dynamic> json
) {
    return _Location.fromJson(
      json
    );
}

/// @nodoc
mixin _$LocationModel {

@JsonKey(name: 'y', fromJson: JsonConverters.toDouble) double get latitude;@JsonKey(name: 'x', fromJson: JsonConverters.toDouble) double get longitude;@JsonKey(name: 'place_name') String? get placeName;@JsonKey(name: 'address_name') String? get addressName;
/// Create a copy of LocationModel
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$LocationModelCopyWith<LocationModel> get copyWith => _$LocationModelCopyWithImpl<LocationModel>(this as LocationModel, _$identity);

  /// Serializes this LocationModel to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is LocationModel&&(identical(other.latitude, latitude) || other.latitude == latitude)&&(identical(other.longitude, longitude) || other.longitude == longitude)&&(identical(other.placeName, placeName) || other.placeName == placeName)&&(identical(other.addressName, addressName) || other.addressName == addressName));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,latitude,longitude,placeName,addressName);

@override
String toString() {
  return 'LocationModel(latitude: $latitude, longitude: $longitude, placeName: $placeName, addressName: $addressName)';
}


}

/// @nodoc
abstract mixin class $LocationModelCopyWith<$Res>  {
  factory $LocationModelCopyWith(LocationModel value, $Res Function(LocationModel) _then) = _$LocationModelCopyWithImpl;
@useResult
$Res call({
@JsonKey(name: 'y', fromJson: JsonConverters.toDouble) double latitude,@JsonKey(name: 'x', fromJson: JsonConverters.toDouble) double longitude,@JsonKey(name: 'place_name') String? placeName,@JsonKey(name: 'address_name') String? addressName
});




}
/// @nodoc
class _$LocationModelCopyWithImpl<$Res>
    implements $LocationModelCopyWith<$Res> {
  _$LocationModelCopyWithImpl(this._self, this._then);

  final LocationModel _self;
  final $Res Function(LocationModel) _then;

/// Create a copy of LocationModel
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? latitude = null,Object? longitude = null,Object? placeName = freezed,Object? addressName = freezed,}) {
  return _then(_self.copyWith(
latitude: null == latitude ? _self.latitude : latitude // ignore: cast_nullable_to_non_nullable
as double,longitude: null == longitude ? _self.longitude : longitude // ignore: cast_nullable_to_non_nullable
as double,placeName: freezed == placeName ? _self.placeName : placeName // ignore: cast_nullable_to_non_nullable
as String?,addressName: freezed == addressName ? _self.addressName : addressName // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}

}


/// Adds pattern-matching-related methods to [LocationModel].
extension LocationModelPatterns on LocationModel {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _Location value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _Location() when $default != null:
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

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _Location value)  $default,){
final _that = this;
switch (_that) {
case _Location():
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _Location value)?  $default,){
final _that = this;
switch (_that) {
case _Location() when $default != null:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function(@JsonKey(name: 'y', fromJson: JsonConverters.toDouble)  double latitude, @JsonKey(name: 'x', fromJson: JsonConverters.toDouble)  double longitude, @JsonKey(name: 'place_name')  String? placeName, @JsonKey(name: 'address_name')  String? addressName)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function(@JsonKey(name: 'y', fromJson: JsonConverters.toDouble)  double latitude, @JsonKey(name: 'x', fromJson: JsonConverters.toDouble)  double longitude, @JsonKey(name: 'place_name')  String? placeName, @JsonKey(name: 'address_name')  String? addressName)  $default,) {final _that = this;
switch (_that) {
case _Location():
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function(@JsonKey(name: 'y', fromJson: JsonConverters.toDouble)  double latitude, @JsonKey(name: 'x', fromJson: JsonConverters.toDouble)  double longitude, @JsonKey(name: 'place_name')  String? placeName, @JsonKey(name: 'address_name')  String? addressName)?  $default,) {final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _Location implements LocationModel {
  const _Location({@JsonKey(name: 'y', fromJson: JsonConverters.toDouble) required this.latitude, @JsonKey(name: 'x', fromJson: JsonConverters.toDouble) required this.longitude, @JsonKey(name: 'place_name') this.placeName, @JsonKey(name: 'address_name') this.addressName});
  factory _Location.fromJson(Map<String, dynamic> json) => _$LocationFromJson(json);

@override@JsonKey(name: 'y', fromJson: JsonConverters.toDouble) final  double latitude;
@override@JsonKey(name: 'x', fromJson: JsonConverters.toDouble) final  double longitude;
@override@JsonKey(name: 'place_name') final  String? placeName;
@override@JsonKey(name: 'address_name') final  String? addressName;

/// Create a copy of LocationModel
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$LocationCopyWith<_Location> get copyWith => __$LocationCopyWithImpl<_Location>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$LocationToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Location&&(identical(other.latitude, latitude) || other.latitude == latitude)&&(identical(other.longitude, longitude) || other.longitude == longitude)&&(identical(other.placeName, placeName) || other.placeName == placeName)&&(identical(other.addressName, addressName) || other.addressName == addressName));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,latitude,longitude,placeName,addressName);

@override
String toString() {
  return 'LocationModel(latitude: $latitude, longitude: $longitude, placeName: $placeName, addressName: $addressName)';
}


}

/// @nodoc
abstract mixin class _$LocationCopyWith<$Res> implements $LocationModelCopyWith<$Res> {
  factory _$LocationCopyWith(_Location value, $Res Function(_Location) _then) = __$LocationCopyWithImpl;
@override @useResult
$Res call({
@JsonKey(name: 'y', fromJson: JsonConverters.toDouble) double latitude,@JsonKey(name: 'x', fromJson: JsonConverters.toDouble) double longitude,@JsonKey(name: 'place_name') String? placeName,@JsonKey(name: 'address_name') String? addressName
});




}
/// @nodoc
class __$LocationCopyWithImpl<$Res>
    implements _$LocationCopyWith<$Res> {
  __$LocationCopyWithImpl(this._self, this._then);

  final _Location _self;
  final $Res Function(_Location) _then;

/// Create a copy of LocationModel
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? latitude = null,Object? longitude = null,Object? placeName = freezed,Object? addressName = freezed,}) {
  return _then(_Location(
latitude: null == latitude ? _self.latitude : latitude // ignore: cast_nullable_to_non_nullable
as double,longitude: null == longitude ? _self.longitude : longitude // ignore: cast_nullable_to_non_nullable
as double,placeName: freezed == placeName ? _self.placeName : placeName // ignore: cast_nullable_to_non_nullable
as String?,addressName: freezed == addressName ? _self.addressName : addressName // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}


}

// dart format on
